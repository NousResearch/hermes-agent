import re

with open("agent/google_adapter.py", "r") as f:
    code = f.read()

# Instead of passing `fc_dict` which is `{ "name": ..., "args": ..., "thought_signature": ... }`,
# Gemini expects `thought_signature` to NOT be inside `functionCall` maybe?
# No, the error says: `Function call is missing a thought_signature in functionCall parts.` 
# This means it literally expects it to be inside `functionCall`.
# But maybe we need to convert everything in the history array to raw dictionaries so that pydantic model_construct doesn't strip it?
# In `build_google_kwargs`, we pass `contents=contents`.
# If `contents` is a list of dicts instead of a list of `types.Content`, the genai SDK might accept it and serialize it raw.
# Oh, we changed `types.Content` to `{"role": ..., "parts": ...}`.
# But `build_google_kwargs` might be missing something? 
# Wait, `fc_dict` was:
# {"name": func_name, "args": func_args, "thought_signature": thought_sig, "thoughtSignature": thought_sig}
# What if it's supposed to be `thoughtSignature`? Yes, Protobufs use camelCase.
# What if it's supposed to be outside `functionCall`? No, the error says `in functionCall parts`.
# What if `google-genai` intercepts the dicts and runs them through `types.Content.model_validate` anyway, which strips unknown fields because `thought_signature` isn't in the schema?
# YES! That's exactly what Pydantic does. `extra='ignore'` is the default.

# To bypass this, we MUST use the underlying REST client OR monkey-patch the Pydantic model.
code = """from google import genai
from google.genai import types
from types import SimpleNamespace
import uuid
import json
import logging

# Monkeypatch types.FunctionCall to allow extra fields so thought_signature survives validation
if hasattr(types.FunctionCall, 'model_config'):
    types.FunctionCall.model_config['extra'] = 'allow'

def build_google_client(api_key: str):
    return genai.Client(api_key=api_key)

def convert_messages_to_google(messages: list):
    contents = []
    system_instruction = None
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_instruction = msg.get("content")
            continue
            
        parts = []
        if isinstance(msg.get("content"), str) and msg["content"].strip():
            parts.append(types.Part.from_text(text=msg["content"]))
        elif isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "text" and part.get("text"):
                    parts.append(types.Part.from_text(text=part["text"]))
        
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                
                extra = tc.get('extra_content', {})
                thought_sig = None
                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_sig = extra["thought_signature"]

                # Use model_construct to bypass init validation, passing thought_signature
                fc = types.FunctionCall.model_construct(name=func_name, args=func_args)
                if thought_sig:
                    fc.thought_signature = thought_sig
                    if not hasattr(fc, "__pydantic_extra__") or fc.__pydantic_extra__ is None:
                        fc.__pydantic_extra__ = {}
                    fc.__pydantic_extra__["thought_signature"] = thought_sig
                    fc.__pydantic_extra__["thoughtSignature"] = thought_sig
                    
                parts.append(types.Part.model_construct(function_call=fc))
                
        if role == "tool":
            parts.append(types.Part.from_function_response(
                name=msg.get("name", "unknown_tool"),
                response={"result": msg.get("content", "")}
            ))
            
        g_role = "user" if role in ("user", "tool") else "model"
        
        if parts:
            contents.append(types.Content(role=g_role, parts=parts))
            
    return contents, system_instruction

def build_google_kwargs(model: str, messages: list, tools: list = None, max_tokens: int = None):
    contents, system_instruction = convert_messages_to_google(messages)
    
    g_tools = []
    if tools:
        for t in tools:
            f = t["function"]
            params = f.get("parameters")
            if params and "type" in params:
                params["type"] = params["type"].upper()
                
            fd = types.FunctionDeclaration(
                name=f["name"],
                description=f.get("description", ""),
                parameters=params
            )
            g_tools.append(fd)
            
    req_model = model.replace("google/", "") if model.startswith("google/") else model
    
    config_kwargs = {"temperature": 0.1}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if max_tokens:
        config_kwargs["max_output_tokens"] = max_tokens
    if g_tools:
        config_kwargs["tools"] = [types.Tool(function_declarations=g_tools)]
        
    config = types.GenerateContentConfig(**config_kwargs)
        
    return {
        "model": req_model,
        "contents": contents,
        "config": config
    }

def normalize_google_response(response, model: str):
    message = SimpleNamespace(role="assistant", content=None, tool_calls=[])
    if response.text:
        message.content = response.text
        
    finish_reason = "stop"
    
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                
                thought_sig = getattr(part.function_call, "thought_signature", getattr(part.function_call, "thoughtSignature", None))
                if not thought_sig and hasattr(part.function_call, "__pydantic_extra__") and part.function_call.__pydantic_extra__:
                    thought_sig = part.function_call.__pydantic_extra__.get("thought_signature") or part.function_call.__pydantic_extra__.get("thoughtSignature")
                
                message.tool_calls.append(SimpleNamespace(
                    id="call_" + str(uuid.uuid4())[:8],
                    type="function",
                    function=SimpleNamespace(
                        name=part.function_call.name,
                        arguments=json.dumps(dict(part.function_call.args))
                    ),
                    extra_content={"thought_signature": thought_sig} if thought_sig else None
                ))
                finish_reason = "tool_calls"

    # Usage tracking
    prompt_tokens = 0
    completion_tokens = 0
    if getattr(response, 'usage_metadata', None):
        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        completion_tokens = response.usage_metadata.candidates_token_count or 0
    
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        model=model,
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, 
            completion_tokens=completion_tokens, 
            total_tokens=prompt_tokens+completion_tokens
        )
    ), finish_reason
"""

with open("agent/google_adapter.py", "w") as f:
    f.write(code)

