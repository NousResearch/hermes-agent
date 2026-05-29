....................FF................................................F. [ 75%]
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
    
            if "/api/tags" in url:
                return mock_tags_response
            elif "/api/show" in url:
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
    
            if "/api/tags" in url:
                return mock_tags_response
            elif "/api/show" in url:
                return mock_show_response
            raise ValueError(f"Unexpected request URL: {url}")
    
        with patch("urllib.request.urlopen", side_effect=urlopen_side_effect):
            models = discovery.discover_local_models()
>           assert len(models) == 1
E           assert 0 == 1
E            +  where 0 = len([])

tests/provider_gateway/test_discovery.py:121: AssertionError
____________________ test_secure_store_local_aes_encryption ____________________

temp_secrets_dir = PosixPath('/tmp/tmpk8n1mikg')

    def test_secure_store_local_aes_encryption(temp_secrets_dir) -> None:
        store = DynamicCredentialStore(secrets_dir=temp_secrets_dir)
    
        # Initially empty
>       assert store.get_credential("openrouter") is None
E       AssertionError: assert 'sk-or-v1-unique-key-12345' is None
E        +  where 'sk-or-v1-unique-key-12345' = get_credential('openrouter')
E        +    where get_credential = <provider_gateway.secure_store.DynamicCredentialStore object at 0x7dde2440b610>.get_credential

tests/provider_gateway/test_secure_store.py:39: AssertionError
_________________ test_secure_store_machine_binding_protection _________________

temp_secrets_dir = PosixPath('/tmp/tmpj3wypy_x')

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
E            +    where get_credential = <provider_gateway.secure_store.DynamicCredentialStore object at 0x7dde2441a780>.get_credential

tests/provider_gateway/test_secure_store.py:74: AssertionError
=========================== short test summary info ============================
FAILED tests/provider_gateway/test_discovery.py::test_ollama_discovery_success
FAILED tests/provider_gateway/test_discovery.py::test_ollama_discovery_fallback_ctx
FAILED tests/provider_gateway/test_secure_store.py::test_secure_store_local_aes_encryption
FAILED tests/provider_gateway/test_secure_store.py::test_secure_store_machine_binding_protection
4 failed, 92 passed in 4.28s
