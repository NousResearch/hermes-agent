import re
with open("agent/google_adapter.py", "r") as f:
    code = f.read()

old = """    return SimpleNamespace(
        id=str(uuid.uuid4()),
        model=model,
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    )"""

new = """    return SimpleNamespace(
        id=str(uuid.uuid4()),
        model=model,
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    ), "stop" """

code = code.replace(old, new)
with open("agent/google_adapter.py", "w") as f:
    f.write(code)
print("Patched return")
