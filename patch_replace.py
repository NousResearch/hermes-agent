import re
with open("agent/google_adapter.py", "r") as f:
    code = f.read()

old = """    if g_tools:
        config.tools = [types.Tool(function_declarations=g_tools)]
        
    return {
        "model": model,
        "contents": contents,"""

new = """    if g_tools:
        config.tools = [types.Tool(function_declarations=g_tools)]
        
    req_model = model.replace("google/", "").replace("google:", "") if model.startswith("google/") or model.startswith("google:") else model
        
    return {
        "model": req_model,
        "contents": contents,"""

code = code.replace(old, new)
with open("agent/google_adapter.py", "w") as f:
    f.write(code)
print("Patched model replacement.")
