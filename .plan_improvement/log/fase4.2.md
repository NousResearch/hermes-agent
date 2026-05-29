....................FF..F.............................................F. [ 75%]
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
            if isinstance(req, str):
                url = req
                req_data = None
            else:
                url = req.full_url
                req_data = req.data
    
            if url.endswith("/api/tags"):
                return mock_tags_response
            elif url.endswith("/api/show"):
                if req_data:
                    payload = json.loads(req_data.decode("utf-8"))
                    if payload.get("name") == "llama3:8b":
                        return mock_show_llama3
                    elif payload.get("name") == "mistral:latest":
                        return mock_show_mistral
                return mock_show_llama3
            raise ValueError(f"Unexpected request URL: {url}")
    
        with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            models = discovery.discover_local_models()
>           assert len(models) == 2
E           assert 0 == 2
E            +  where 0 = len([])

tests/provider_gateway/test_discovery.py:74: AssertionError
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
            if isinstance(req, str):
                url = req
            else:
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

tests/provider_gateway/test_discovery.py:121: AssertionError
____________________ test_server_chat_completions_streaming ____________________

mock_create = <MagicMock name='create' id='139560605805472'>
running_server = (53873, <http.server.ThreadingHTTPServer object at 0x7eedfecf2900>)

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
    
        with urllib.request.urlopen(req, timeout=2.0) as response:
            assert response.status == 200
>           lines = response.read().decode("utf-8").splitlines()
                    ^^^^^^^^^^^^^^^

tests/provider_gateway/test_gateway_server.py:129: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/http/client.py:492: in read
    s = self.fp.read()
        ^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <socket.SocketIO object at 0x7eedff5836d0>
b = bytearray(b'data: {"choices": [{"delta": {"content": " world!"}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n\x00\x0...0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')

    def readinto(self, b):
        """Read up to len(b) bytes into the writable buffer *b* and return
        the number of bytes read.  If the socket is non-blocking and no bytes
        are available, None is returned.
    
        If *b* is non-empty, a 0 return value indicates that the connection
        was shutdown at the other end.
        """
        self._checkClosed()
        self._checkReadable()
        if self._timeout_occurred:
            raise OSError("cannot read from timed out object")
        try:
>           return self._sock.recv_into(b)
                   ^^^^^^^^^^^^^^^^^^^^^^^
E           TimeoutError: timed out

../../../.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/lib/python3.14/socket.py:725: TimeoutError
____________________ test_secure_store_local_aes_encryption ____________________

temp_secrets_dir = PosixPath('/tmp/tmpxfae4s_p')

    def test_secure_store_local_aes_encryption(temp_secrets_dir) -> None:
        store = DynamicCredentialStore(secrets_dir=temp_secrets_dir)
    
        # Initially empty
>       assert store.get_credential("openrouter") is None
E       AssertionError: assert 'sk-or-v1-unique-key-12345' is None
E        +  where 'sk-or-v1-unique-key-12345' = get_credential('openrouter')
E        +    where get_credential = <provider_gateway.secure_store.DynamicCredentialStore object at 0x7eedfc40b610>.get_credential

tests/provider_gateway/test_secure_store.py:21: AssertionError
_________________ test_secure_store_machine_binding_protection _________________

temp_secrets_dir = PosixPath('/tmp/tmp6jiunllj')

    def test_secure_store_machine_binding_protection(temp_secrets_dir) -> None:
        """Verify that credentials cannot be decrypted if the hardware fingerprint changes (machine-bound)."""
        store = DynamicCredentialStore(secrets_dir=temp_secrets_dir)
        api_key = "extremely-sensitive-token"
    
        # Force keyring import failure to isolate local AES fallback testing
        with patch("builtins.__import__", side_effect=ImportError("No keyring")):
            # Store with current fingerprint
>           assert store.store_credential("cohere", api_key) is True
E           AssertionError: assert False is True
E            +  where False = store_credential('cohere', 'extremely-sensitive-token')
E            +    where store_credential = <provider_gateway.secure_store.DynamicCredentialStore object at 0x7eedfc417a80>.store_credential

tests/provider_gateway/test_secure_store.py:53: AssertionError
------------------------------ Captured log call -------------------------------
ERROR    provider_gateway.secure_store:secure_store.py:107 Failed to store credentials using local fallback encryption: No keyring
=========================== short test summary info ============================
FAILED tests/provider_gateway/test_discovery.py::test_ollama_discovery_success
FAILED tests/provider_gateway/test_discovery.py::test_ollama_discovery_fallback_ctx
FAILED tests/provider_gateway/test_gateway_server.py::test_server_chat_completions_streaming
FAILED tests/provider_gateway/test_secure_store.py::test_secure_store_local_aes_encryption
FAILED tests/provider_gateway/test_secure_store.py::test_secure_store_machine_binding_protection
5 failed, 91 passed in 5.70s
