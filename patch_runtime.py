with open("hermes_cli/runtime_provider.py", "r") as f:
    code = f.read()

old_block = """        if provider == "copilot":
            api_mode = _copilot_runtime_api_mode(model_cfg, creds.get("api_key", ""))
        else:"""

new_block = """        if provider == "google":
            api_mode = "google_genai"
        elif provider == "copilot":
            api_mode = _copilot_runtime_api_mode(model_cfg, creds.get("api_key", ""))
        else:"""

if old_block in code:
    code = code.replace(old_block, new_block)
    with open("hermes_cli/runtime_provider.py", "w") as f:
        f.write(code)
    print("Patched runtime_provider.py")
else:
    print("Failed to patch runtime_provider.py")
