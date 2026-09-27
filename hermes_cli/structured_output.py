"""Local JSON Schema contract for one-shot CLI answers (not constrained decoding)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import os
import tempfile
import stat


def _check_output_target(path):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise ValueError(f"Output file must be a regular file, not a symlink or special file: {path}")


def prepare_output_path(path):
    path = Path(path).expanduser().absolute()
    try:
        # Resolve the parent only: resolving the final component would follow an output symlink.
        path = path.parent.resolve(strict=True) / path.name
        _check_output_target(path)
    except OSError as exc:
        raise ValueError(f"Cannot use output file {path}: {exc}") from exc
    return path


def write_last_message(path, text):
    """Stage in the destination directory so publication is one atomic replacement."""
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _check_output_target(path)
        os.replace(temporary, path)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot write final output file {path}: {exc}") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _finite_number(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite JSON number: {text}")
    return value


def strict_json_loads(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=_finite_number, parse_float=_finite_number)


class OutputSchema:
    def __init__(self, schema, validator):
        self.schema = schema
        self.validator = validator

    @classmethod
    def load(cls, path):
        try:
            from jsonschema import Draft202012Validator, SchemaError
            from jsonschema.validators import validator_for
            from referencing import Registry, Resource
            from referencing.exceptions import Unresolvable
            from referencing.jsonschema import specification_with
        except ImportError as exc:
            raise ValueError("Output schema validation requires jsonschema; repair the Hermes environment with 'hermes pm repair'.") from exc

        try:
            schema = strict_json_loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
            if not isinstance(schema, (dict, bool)):
                raise ValueError("JSON Schema must be an object or boolean")
            validator_cls = validator_for(schema, default=Draft202012Validator if not isinstance(schema, dict) or "$schema" not in schema else None)
            if validator_cls is None:
                raise ValueError(f"Unsupported JSON Schema dialect: {schema['$schema']}")
            validator_cls.check_schema(schema)
            dialect = validator_cls.META_SCHEMA.get("$id", validator_cls.META_SCHEMA.get("id"))
            resource = Resource.from_contents(schema, default_specification=specification_with(dialect))
            registry = Registry().with_resource("", resource).crawl()
            seen = set()

            def check_refs(current, resolver):
                node = current.contents
                if id(node) in seen:
                    return
                seen.add(id(node))
                if isinstance(node, dict):
                    if "$schema" in node and validator_for(node, default=None) is None:
                        raise ValueError(f"Unsupported JSON Schema dialect: {node['$schema']}")
                    node_validator = validator_for(node, default=validator_cls)
                    supported_vocabularies = node_validator.META_SCHEMA.get("$vocabulary", {})
                    for uri, required in node.get("$vocabulary", {}).items():
                        if required and uri not in supported_vocabularies:
                            raise ValueError(f"Unsupported required schema vocabulary: {uri}")
                    for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
                        if keyword in node:
                            ref = node[keyword]
                            if not isinstance(ref, str) or not ref.startswith("#"):
                                raise ValueError(f"Only local fragment references are supported: {ref!r}")
                            try:
                                resolved = resolver.lookup(ref)
                                if not isinstance(resolved.contents, (dict, bool)):
                                    raise ValueError("Referenced JSON Schema must be an object or boolean")
                                node_validator.check_schema(resolved.contents)
                                check_refs(Resource.from_contents(resolved.contents,
                                           default_specification=specification_with(dialect)), resolved.resolver)
                            except Unresolvable as exc:
                                raise ValueError(f"Invalid local schema reference: {ref!r}") from exc
                for child in current.subresources():
                    check_refs(child, resolver.in_subresource(child))

            check_refs(resource, registry.resolver_with_root(resource))
            return cls(schema, validator_cls(schema, registry=registry))
        except (OSError, UnicodeError, ValueError, TypeError, SchemaError) as exc:
            raise ValueError(f"Cannot use output schema {path}: {exc}") from exc

    def instruct(self, query):
        instruction = (
            "\n\nReturn your final answer as one JSON value matching this JSON Schema. "
            "Do not include markdown fences or any prose outside the JSON.\n"
            + json.dumps(self.schema, ensure_ascii=False)
        )
        if isinstance(query, list):
            return [*query, {"type": "text", "text": instruction}]
        return (query or "") + instruction

    def validate(self, text):
        from jsonschema import ValidationError
        from referencing.exceptions import Unresolvable
        try:
            value = strict_json_loads(text)
            self.validator.validate(value)
            return value
        except ValidationError as exc:
            raise ValueError(f"{exc.json_path}: {exc.message}") from exc
        except (Unresolvable, RecursionError) as exc:
            raise ValueError(f"Schema validation could not complete: {exc}") from exc

    def apply(self, result, agent):
        errors = []
        for attempt in range(2):
            try:
                value = self.validate(result.get("final_response", ""))
                return {**result, "structured_output": value}
            except ValueError as exc:
                errors.append(str(exc))
            if attempt == 0:
                try:
                    result = _correct_response(agent, result, self, errors[-1])
                    if result.get("failed"):
                        errors.append(result["error"])
                        break
                except (KeyboardInterrupt, InterruptedError):
                    return {**result, "failed": True, "completed": False, "interrupted": True,
                            "error": "Interrupted during output schema correction", "schema_errors": errors}
                except Exception as exc:
                    errors.append(f"Correction failed: {type(exc).__name__}: {exc}")
                    break
        return {**result,
                "failed": True, "completed": False, "failure_reason": "output_schema",
                "error": f"Final response failed output schema validation: {errors[-1]}",
                "schema_errors": errors}


def _correct_response(agent, result, schema, error):
    """One native completion, never another agent/tool loop. Keep tool schemas for cache reuse."""
    import logging
    from agent.conversation_loop import _moa_client_consumes_prepared_request
    from agent.turn_request_assembly import assemble_api_request
    from agent.message_sanitization import sanitize_outbound_kwargs, strip_images_for_rejecting_model
    from agent.message_metadata import append_message
    from agent.turn_truncation import normalize_response_for_agent

    import time
    if getattr(agent, "_interrupt_requested", False):
        raise InterruptedError("Interrupted before output schema correction")
    budget = getattr(agent, "run_budget_seconds", None)
    started = getattr(agent, "_run_budget_started_at", None)
    if budget and started is not None and time.time() - started >= budget:
        raise ValueError("No run budget remains for output schema correction")
    iterations = getattr(agent, "iteration_budget", None)
    if iterations is not None and iterations.remaining <= 0:
        raise ValueError("No iteration budget remains for output schema correction")

    history = result.get("messages")
    if not history or history[-1].get("role") != "assistant":
        raise ValueError("Cannot correct without the completed original transcript")
    nudge = {"role": "user", "content": schema.instruct(
        "Correct only the JSON formatting/schema of your last answer using the existing transcript. "
        "Do not repeat the task or call tools; prior actions have already happened. "
        f"Validation error: {error}"
    )}
    messages = list(history)
    append_message(messages, nudge)
    # Reuse the turn's frozen prompt and canonical wire/cache preparation, not the
    # summary path (which omits the split system prefix and cache breakpoints).
    assembled = assemble_api_request(
        agent, messages=messages, current_turn_user_idx=result.get("current_turn_user_idx", -1),
        _ext_prefetch_cache=None, _plugin_user_context=None, moa_config=None,
        active_system_prompt=agent._cached_system_prompt, original_user_message=nudge["content"],
        pending_moa_prepared_request=None, request_logger=logging.getLogger(__name__),
    )
    strip_images_for_rejecting_model(agent, assembled.api_messages)
    request = agent._build_api_kwargs(assembled.api_messages, tools_for_api=assembled.tools_for_api)
    sanitize_outbound_kwargs(agent, request)
    # Assembly already ran the advisors; only the live MoA facade consumes this handshake.
    if (assembled._moa_prepared_request is not None and agent.provider == "moa"
            and _moa_client_consumes_prepared_request(agent.client)):
        request["_moa_prepared_request"] = assembled._moa_prepared_request
    started_call = time.monotonic()
    response = agent._interruptible_api_call(request)
    from agent.turn_usage import record_response_usage
    record_response_usage(agent, response, messages=messages, api_call_count=2,
                          api_duration=time.monotonic() - started_call,
                          compression_attempts=0, max_compression_attempts=0)
    normalized = normalize_response_for_agent(agent, response)
    from agent.usage_pricing import normalize_usage
    usage = normalize_usage(getattr(response, "usage", None), provider=agent.provider, api_mode=agent.api_mode)
    result = dict(result)
    for key in ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens"):
        result[key] = (result.get(key) or 0) + getattr(usage, key)
    text = normalized.content or ""
    result["final_response"] = text
    if normalized.tool_calls or normalized.finish_reason not in {"stop", "end_turn"}:
        result.update(failed=True, completed=False,
                      error=f"Correction did not finish with a final answer ({normalized.finish_reason}); no tools were executed.")
        return result
    # Use the same metadata-preserving storage boundary as an ordinary final
    # answer; adopting the transcript also makes the one-shot exit flush retry it.
    append_message(messages, agent._build_assistant_message(normalized, normalized.finish_reason))
    agent._persist_session(messages)
    result["messages"] = messages
    return result
