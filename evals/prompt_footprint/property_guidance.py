"""Offline prompt A/B and real optional catalog -> skill_view probe.
Run with the repository venv; tiktoken must be available. No model API calls.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    with tempfile.TemporaryDirectory(prefix="property-guidance-") as temporary:
        os.environ["HERMES_HOME"] = temporary
        os.environ["TERMINAL_CWD"] = temporary
        os.chdir(temporary)
        import tiktoken
        from agent import prompt_builder
        from agent.system_prompt import build_system_prompt
        from tools.skills_hub_official import OptionalSkillSource
        from tools.skills_tool import skill_view

        agent = SimpleNamespace(
            load_soul_identity=False, skip_context_files=True, valid_tool_names=[],
            _task_completion_guidance=False, _tool_use_enforcement=False,
            _environment_probe=False, _kanban_worker_guidance="", _memory_store=None,
            _memory_manager=None, model="", provider="", platform="desktop",
            pass_session_id=False, session_id="", _emit_status=lambda *args: None,
        )
        fixed_hint = prompt_builder.PLATFORM_HINTS["desktop"]
        # The original clause is taken from the pinned base, never synthesized.
        source = subprocess.check_output(
            ["git", "show", f"{sys.argv[1]}:agent/prompt_builder.py"], cwd=ROOT,
            text=True, stdin=subprocess.DEVNULL,
        )
        import ast
        tree = ast.parse(source)
        mapping = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "PLATFORM_HINTS" for t in n.targets))
        base_hint = next(ast.literal_eval(value) for key, value in zip(mapping.keys, mapping.values)
                         if isinstance(key, ast.Constant) and key.value == "desktop")
        prompt_builder.PLATFORM_HINTS["desktop"] = base_hint
        before = build_system_prompt(agent)
        prompt_builder.PLATFORM_HINTS["desktop"] = fixed_hint
        after = build_system_prompt(agent)
        assert before.replace(base_hint, fixed_hint) == after
        result = {"base": sys.argv[1], "tokenizers": {}}
        for name in ("cl100k_base", "o200k_base"):
            enc = tiktoken.get_encoding(name)
            counts = [len(enc.encode(s)) for s in (base_hint, fixed_hint, before, after)]
            result["tokenizers"][name] = dict(zip(
                ("hint_before", "hint_after", "prompt_before", "prompt_after"), counts))
        optional = OptionalSkillSource()
        matches = [m for m in optional.list_local() if "property" in m.tags and "rental" in m.tags]
        assert matches
        bundle = optional.fetch(matches[0].identifier)
        assert bundle
        dest = Path(temporary) / "skills" / bundle.name
        dest.mkdir(parents=True)
        for name, data in bundle.files.items():
            (dest / name).write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        loaded = json.loads(skill_view(bundle.name))
        assert loaded["success"], loaded
        example = loaded["content"].split("```listing\n", 1)[1].split("```", 1)[0]
        assert json.loads(example)["address"]
        result.update(identifier=matches[0].identifier, skill_view_success=True,
                      non_property_prompt_byte_parity=True, example=json.loads(example))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
