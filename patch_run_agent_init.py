import re
with open("run_agent.py", "r") as f:
    code = f.read()

# 1. Update api_mode handling in __init__
old_api_mode_init = """        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages"}:
            self.api_mode = api_mode"""
new_api_mode_init = """        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages", "google_genai"}:
            self.api_mode = api_mode"""
code = code.replace(old_api_mode_init, new_api_mode_init)

# 2. Add build_google_kwargs to _build_api_kwargs
old_build_kwargs = """    def _build_api_kwargs(self, api_messages: list) -> dict:
        \"\"\"Build the keyword arguments dict for the active API mode.\"\"\"
        if self.api_mode == "anthropic_messages":"""
new_build_kwargs = """    def _build_api_kwargs(self, api_messages: list) -> dict:
        \"\"\"Build the keyword arguments dict for the active API mode.\"\"\"
        if self.api_mode == "google_genai":
            from agent.google_adapter import build_google_kwargs
            return build_google_kwargs(
                model=self.model,
                messages=api_messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
            )

        if self.api_mode == "anthropic_messages":"""
code = code.replace(old_build_kwargs, new_build_kwargs)

# 3. Add Google client initialization to _init_agent
old_init_client = """        self._anthropic_client = None
        self._is_anthropic_oauth = False

        if self.api_mode == "anthropic_messages":"""
new_init_client = """        self._anthropic_client = None
        self._is_anthropic_oauth = False
        self._google_client = None

        if self.api_mode == "google_genai":
            from agent.google_adapter import build_google_client
            self._google_client = build_google_client(api_key=api_key)
            self.client = None
            self._client_kwargs = {}
            if not self.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {self.model} (Google Native)")

        if self.api_mode == "anthropic_messages":"""
code = code.replace(old_init_client, new_init_client)

# 4. Add google_genai handling to _interruptible_api_call
old_api_call = """                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)"""
new_api_call = """                elif self.api_mode == "google_genai":
                    response = self._google_client.models.generate_content(**api_kwargs)
                    from agent.google_adapter import normalize_google_response
                    result["response"], _ = normalize_google_response(response, self.model)
                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)"""
code = code.replace(old_api_call, new_api_call)

# 5. Fallback model refresh handles Google client
old_fb = """                self._anthropic_client = build_anthropic_client(effective_key, self._anthropic_base_url)
                self._is_anthropic_oauth = _is_oauth_token(effective_key)
                self.client = None
                self._client_kwargs = {}"""
new_fb = """                self._anthropic_client = build_anthropic_client(effective_key, self._anthropic_base_url)
                self._is_anthropic_oauth = _is_oauth_token(effective_key)
                self.client = None
                self._client_kwargs = {}
            elif fb_api_mode == "google_genai":
                from agent.google_adapter import build_google_client
                self.api_key = fb_client.api_key
                self._google_client = build_google_client(api_key=self.api_key)
                self.client = None
                self._client_kwargs = {}"""
code = code.replace(old_fb, new_fb)

# 6. Normalize response handler in `chat` or main loop
old_norm = """                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import normalize_anthropic_response
                    assistant_message, finish_reason = normalize_anthropic_response(
                        response, strip_tool_prefix=self._is_anthropic_oauth
                    )"""
new_norm = """                elif self.api_mode == "google_genai":
                    # For non-streaming fallback, the normalization is already handled in _interruptible_api_call
                    # because it returns the SimpleNamespace wrapper. So we just unwrap it identically to chat_completions.
                    assistant_message = response.choices[0].message
                    finish_reason = response.choices[0].finish_reason
                    self._update_token_usage(response.usage)
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import normalize_anthropic_response
                    assistant_message, finish_reason = normalize_anthropic_response(
                        response, strip_tool_prefix=self._is_anthropic_oauth
                    )"""
code = code.replace(old_norm, new_norm)

with open("run_agent.py", "w") as f:
    f.write(code)

print("run_agent.py patched.")
