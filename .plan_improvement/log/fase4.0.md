....................FF.FF..F............................................ [ 75%]
F.......................                                                 [100%]
=================================== FAILURES ===================================
________________________ test_ollama_discovery_success _________________________

    def test_ollama_discovery_success() -> None:
        """Test successful discovery of Ollama models with context length resolution."""
        discovery = OllamaDiscovery(host="http://127.0.0.1:11434")
    
        # Mock response for /api/tags
        mock_tags_response = MagicMock()
        mock_tags_response.status = 200
        mock_tags_response.read.return_value = json.dumps({
            "models": [
                {"name": "llama3:8b"},
                {"name": "mistral:latest"},
            ]
        }).encode("utf-8")
    
        # Mock responses for /api/show
        mock_show_llama3 = MagicMock()
        mock_show_llama3.status = 200
        mock_show_llama3.read.return_value = json.dumps({
            "parameters": "num_ctx        8192\nstop           <|im_end|>",
            "model_info": {
                "llama.context_length": 8192
            }
        }).encode("utf-8")
    
        mock_show_mistral = MagicMock()
        mock_show_mistral.status = 200
        # Test fallback model_info context length parsing
        mock_show_mistral.read.return_value = json.dumps({
            "parameters": "",
            "model_info": {
                "mistral.context_length": 32768
            }
        }).encode("utf-8")
    
        # urllib.request.urlopen side effect
        def urlopen_side_effect(req, timeout=1.0):
            url = req.full_url
            if url.endswith("/api/tags"):
                return mock_tags_response
            elif url.endswith("/api/show"):
                payload = json.loads(req.data.decode("utf-8"))
                if payload.get("name") == "llama3:8b":
                    return mock_show_llama3
                elif payload.get("name") == "mistral:latest":
                    return mock_show_mistral
            raise ValueError(f"Unexpected request URL: {url}")
    
        with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            models = discovery.discover_local_models()
>           assert len(models) == 2
E           assert 0 == 2
E            +  where 0 = len([])

tests/provider_gateway/test_discovery.py:66: AssertionError
______________________ test_ollama_discovery_fallback_ctx ______________________

    def test_ollama_discovery_fallback_ctx() -> None:
        """Test fallback context length logic."""
        discovery = OllamaDiscovery(host="http://127.0.0.1:11434")
    
        mock_tags_response = MagicMock()
        mock_tags_response.status = 200
        mock_tags_response.read.return_value = json.dumps({
            "models": [{"name": "unknown-model:latest"}]
        }).encode("utf-8")
    
        # Failing API show or no ctx information
        mock_show_response = MagicMock()
        mock_show_response.status = 200
        mock_show_response.read.return_value = json.dumps({
            "parameters": "",
            "model_info": {}
        }).encode("utf-8")
    
        def urlopen_side_effect(req, timeout=1.0):
            url = req.full_url
            if url.endswith("/api/tags"):
                return mock_tags_response
            elif url.endswith("/api/show"):
                return mock_show_response
            raise ValueError(f"Unexpected request URL: {url}")
    
        with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            models = discovery.discover_local_models()
>           assert len(models) == 1
E           assert 0 == 1
E            +  where 0 = len([])

tests/provider_gateway/test_discovery.py:109: AssertionError
__________________ test_server_chat_completions_non_streaming __________________

mock_create = <MagicMock name='create' id='137699744984640'>
running_server = (56939, <http.server.ThreadingHTTPServer object at 0x7d3cbac32a50>)

    @patch("openai.resources.chat.completions.Completions.create")
    def test_server_chat_completions_non_streaming(mock_create, running_server) -> None:
        port, _ = running_server
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
    
        # Setup mock response
        mock_choice = MagicMock()
        mock_choice.message = MagicMock(content="Hello from mock server!")
        mock_choice.message.role = "assistant"
        mock_choice.finish_reason = "stop"
    
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "id": "chat-mock-123",
            "object": "chat.completion",
            "created": 1677610602,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from mock server!"
                },
                "finish_reason": "stop"
            }]
        }
        mock_create.return_value = mock_resp
    
        payload = json.dumps({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False
        }).encode("utf-8")
    
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
    
>       with urllib.request.urlopen(req, timeout=2.0) as response:
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_gateway_server.py:83: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:187: in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:493: in open
    response = meth(req, response)
               ^^^^^^^^^^^^^^^^^^^
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:602: in http_response
    response = self.parent.error(
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:531: in error
    return self._call_chain(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:464: in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <urllib.request.HTTPDefaultErrorHandler object at 0x7d3cbac338c0>
req = <urllib.request.Request object at 0x7d3cb86ef530>
fp = <http.client.HTTPResponse object at 0x7d3cb8936320>, code = 502
msg = 'Bad Gateway', hdrs = <http.client.HTTPMessage object at 0x7d3cb85f3570>

    def http_error_default(self, req, fp, code, msg, hdrs):
>       raise HTTPError(req.full_url, code, msg, hdrs, fp)
E       urllib.error.HTTPError: HTTP Error 502: Bad Gateway

../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:611: HTTPError
____________________ test_server_chat_completions_streaming ____________________

mock_create = <MagicMock name='create' id='137699744986992'>
running_server = (56939, <http.server.ThreadingHTTPServer object at 0x7d3cbac32a50>)

    @patch("openai.resources.chat.completions.Completions.create")
    def test_server_chat_completions_streaming(mock_create, running_server) -> None:
        port, _ = running_server
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
    
        # Setup mock streaming chunks
        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }
    
        chunk2 = MagicMock()
        chunk2.model_dump.return_value = {
            "choices": [{
                "delta": {"content": " world!"},
                "finish_reason": "stop"
            }]
        }
    
        mock_create.return_value = [chunk1, chunk2]
    
        payload = json.dumps({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True
        }).encode("utf-8")
    
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
    
>       with urllib.request.urlopen(req, timeout=2.0) as response:
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/provider_gateway/test_gateway_server.py:127: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:187: in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:493: in open
    response = meth(req, response)
               ^^^^^^^^^^^^^^^^^^^
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:602: in http_response
    response = self.parent.error(
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:531: in error
    return self._call_chain(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:464: in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <urllib.request.HTTPDefaultErrorHandler object at 0x7d3cbac338c0>
req = <urllib.request.Request object at 0x7d3cb85f3bd0>
fp = <http.client.HTTPResponse object at 0x7d3cb8a2cd90>, code = 502
msg = 'Bad Gateway', hdrs = <http.client.HTTPMessage object at 0x7d3cb861da90>

    def http_error_default(self, req, fp, code, msg, hdrs):
>       raise HTTPError(req.full_url, code, msg, hdrs, fp)
E       urllib.error.HTTPError: HTTP Error 502: Bad Gateway

../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/urllib/request.py:611: HTTPError
__________________ test_streaming_deanonimizer_sliding_buffer __________________

    def test_streaming_deanonimizer_sliding_buffer() -> None:
        """Test real-time de-anonymization of sliding buffer chunks."""
        sanitizer = PIISanitizer()
        prompt = "Tolong analisis email void@example.com."
        sanitized = sanitizer.sanitize_prompt(prompt)
        placeholder = "[REDACTED_EMAIL_1]"
        assert placeholder in sanitized
    
        deanonimizer = sanitizer.get_deanonimizer()
    
        # Simulate stream chunks coming in
        chunk1 = "Hasil untuk "
        out1 = deanonimizer.process_chunk(chunk1)
        assert out1 == "Hasil untuk "
    
        # Chunk splitting the placeholder: "[REDACTED_EM"
        chunk2 = "[REDACTED_EM"
        out2 = deanonimizer.process_chunk(chunk2)
        assert out2 == ""  # should hold back the open bracket part!
    
        # Next chunk completing the placeholder: "AIL_1] aman."
        chunk3 = "AIL_1] aman."
        out3 = deanonimizer.process_chunk(chunk3)
>       assert out3 == "void@example.com aman."
E       AssertionError: assert 'void@example.com. aman.' == 'void@example.com aman.'
E         
E         - void@example.com aman.
E         + void@example.com. aman.
E         ?                 +

tests/provider_gateway/test_guardrails.py:67: AssertionError
_________________ test_secure_store_machine_binding_protection _________________

temp_secrets_dir = PosixPath('/tmp/tmpig9t09p6')

    def test_secure_store_machine_binding_protection(temp_secrets_dir) -> None:
        """Verify that credentials cannot be decrypted if the hardware fingerprint changes (machine-bound)."""
        store = DynamicCredentialStore(secrets_dir=temp_secrets_dir)
        api_key = "extremely-sensitive-token"
    
        # Store with current fingerprint
        assert store.store_credential("cohere", api_key) is True
    
        # Mock the fingerprint method to simulate another machine trying to decrypt the file
        with patch.object(store, "_get_machine_fingerprint", return_value=b"totally_different_machine_fingerprint_123"):
            # Decryption should fail and return None
>           assert store.get_credential("cohere") is None
E           AssertionError: assert 'extremely-sensitive-token' is None
E            +  where 'extremely-sensitive-token' = get_credential('cohere')
E            +    where get_credential = <provider_gateway.secure_store.DynamicCredentialStore object at 0x7d3cb81d3390>.get_credential

tests/provider_gateway/test_secure_store.py:56: AssertionError
=========================== short test summary info ============================
FAILED tests/provider_gateway/test_discovery.py::test_ollama_discovery_success
FAILED tests/provider_gateway/test_discovery.py::test_ollama_discovery_fallback_ctx
FAILED tests/provider_gateway/test_gateway_server.py::test_server_chat_completions_non_streaming
FAILED tests/provider_gateway/test_gateway_server.py::test_server_chat_completions_streaming
FAILED tests/provider_gateway/test_guardrails.py::test_streaming_deanonimizer_sliding_buffer
FAILED tests/provider_gateway/test_secure_store.py::test_secure_store_machine_binding_protection
6 failed, 90 passed in 15.86s
