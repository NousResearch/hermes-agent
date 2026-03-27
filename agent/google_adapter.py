from google import genai
from google.genai import types
from types import SimpleNamespace
import uuid
import json

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
        if isinstance(msg.get("content"), str):
            parts.append(types.Part.from_text(text=msg["content"]))
        elif isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "text":
                    parts.append(types.Part.from_text(text=part["text"]))
                    
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])
                parts.append(types.Part.from_function_call(name=func_name, args=func_args))
                
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
            props = f["parameters"].get("properties", {})
            req = f["parameters"].get("required", [])
            g_tools.append(types.FunctionDeclaration(
                name=f["name"],
                description=f.get("description", ""),
                # Simplified schema translation
            ))
            
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_tokens,
        temperature=0.1
    )
    if g_tools:
        config.tools = [types.Tool(function_declarations=g_tools)]
        
    return {
        "model": model,
        "contents": contents,
        "config": config
    }

def normalize_google_response(response, model: str):
    message = SimpleNamespace(role="assistant", content=None, tool_calls=[])
    if response.text:
        message.content = response.text
        
    for part in response.candidates[0].content.parts:
        if part.function_call:
            message.tool_calls.append(SimpleNamespace(
                id="call_" + str(uuid.uuid4())[:8],
                type="function",
                function=SimpleNamespace(
                    name=part.function_call.name,
                    arguments=json.dumps(dict(part.function_call.args))
                )
            ))
            
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        model=model,
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    )
