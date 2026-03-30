import re
with open("run_agent.py", "r") as f:
    code = f.read()

google_stream_block = """        def _call_google():
            has_tool_use = False
            full_text = []
            
            response = self._google_client.models.generate_content_stream(**api_kwargs)
            for chunk in response:
                if self._interrupt_requested:
                    break
                    
                last_chunk_time["t"] = time.time()
                
                # Check for tool calls
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if part.function_call:
                            has_tool_use = True
                            tool_name = part.function_call.name
                            if tool_name:
                                _fire_first_delta()
                                self._fire_tool_gen_started(tool_name)
                                
                if chunk.text and not has_tool_use:
                    _fire_first_delta()
                    self._fire_stream_delta(chunk.text)
                    full_text.append(chunk.text)
                    deltas_were_sent["yes"] = True
                    
            from agent.google_adapter import normalize_google_response
            # The iterator also holds the final aggregated response object in many SDKs, or we can just pass the last chunk
            # Actually, google-genai generation stream object is iterable but doesn't have a get_final_message()
            # For simplicity in this patch, we'll just run non-streaming if it's too complex to reconstruct, 
            # OR we can just use the final chunk. The final chunk contains the full merged Usage metadata.
            # But the full text isn't in the final chunk, we have to rebuild it.
            # Let's just fallback to non-streaming for Google in this PR to be safe, or build the mock.
            # Actually, _interruptible_api_call handles non-streaming.
            pass
"""

# Let's just inject the google_genai call into _call()
old_call = """                        if self.api_mode == "anthropic_messages":
                            self._try_refresh_anthropic_client_credentials()
                            result["response"] = _call_anthropic()
                        else:
                            result["response"] = _call_chat_completions()"""

new_call = """                        if self.api_mode == "anthropic_messages":
                            self._try_refresh_anthropic_client_credentials()
                            result["response"] = _call_anthropic()
                        elif self.api_mode == "google_genai":
                            # Streaming with google-genai is complex to mock back to OpenRouter shape,
                            # so we fallback to non-streaming native call which already normalizes correctly.
                            res = self._google_client.models.generate_content(**api_kwargs)
                            from agent.google_adapter import normalize_google_response
                            result["response"], _ = normalize_google_response(res, self.model)
                            # Fire a single delta so the UI knows it finished
                            if res.text:
                                _fire_first_delta()
                                self._fire_stream_delta(res.text)
                        else:
                            result["response"] = _call_chat_completions()"""

if old_call in code:
    code = code.replace(old_call, new_call)
    with open("run_agent.py", "w") as f:
        f.write(code)
    print("Patched run_agent.py stream block")
