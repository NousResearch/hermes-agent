import re
with open("run_agent.py", "r") as f:
    code = f.read()

old_kwargs = """        api_kwargs = {
            "model": self.model,
            "messages": sanitized_messages,
            "tools": self.tools if self.tools else None,"""

new_kwargs = """        # Gemini AI Studio endpoint rejects models with the "google/" prefix
        _req_model = self.model
        if "generativelanguage.googleapis.com" in self._base_url_lower:
            _req_model = _req_model.replace("google/", "")

        api_kwargs = {
            "model": _req_model,
            "messages": sanitized_messages,
            "tools": self.tools if self.tools else None,"""

if old_kwargs in code:
    code = code.replace(old_kwargs, new_kwargs)
    with open("run_agent.py", "w") as f:
        f.write(code)
    print("Patched run_agent.py!")
else:
    print("Could not find block to patch")
