import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from tinker.types import SampleResponse, SampledSequence


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.base_model = "meta-llama/Llama-3.1-8B-Instruct"

    trainer.tokenizer = MagicMock()
    trainer.tokenizer.encode = MagicMock(side_effect=lambda text, **kwargs: [1, 2, 3, 4, 5])
    trainer.tokenizer.decode = MagicMock(return_value="Test output")
    trainer.tokenizer.apply_chat_template = MagicMock(
        return_value="System: You are helpful\nUser: Hello"
    )

    trainer.current_sampling_client = MagicMock()

    mock_sequence = SampledSequence(
        tokens=[10, 11, 12],
        logprobs=[0.9, 0.8, 0.7],
        stop_reason="stop",
    )
    mock_result = SampleResponse(sequences=[mock_sequence])

    trainer.current_sampling_client.sample_async = AsyncMock(return_value=mock_result)

    return trainer


@pytest.fixture
def app_with_trainer(mock_trainer):
    import tinker_atropos.trainer as trainer_module

    original_trainer = trainer_module.trainer
    trainer_module.trainer = mock_trainer

    # FastAPI TestClient needs an instance of the application as an input parameter
    yield trainer_module.app

    trainer_module.trainer = original_trainer


@pytest.fixture
def client(app_with_trainer):
    return TestClient(app_with_trainer)


class TestHealthEndpoint:
    def test_health_with_trainer(self, client):
        """Test health endpoint when trainer is initialized"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["trainer_initialized"] is True


class TestCompletionsEndpoint:
    def test_single_completion(self, client, mock_trainer):
        """Test single completion request"""
        request_data = {
            "prompt": "Once upon a time",
            "max_tokens": 100,
            "temperature": 0.7,
            "n": 1,
        }

        response = client.post("/v1/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert data["id"].startswith("cmpl-")
        assert data["model"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["text"] == "Test output"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_multiple_completions(self, client, mock_trainer):
        """Test requesting multiple completions (n > 1)"""
        # Update mock to return multiple sequences
        mock_sequences = [
            SampledSequence(tokens=[10, 11], logprobs=[0.9, 0.8], stop_reason="stop"),
            SampledSequence(tokens=[12, 13], logprobs=[0.7, 0.6], stop_reason="stop"),
        ]
        mock_result = SampleResponse(sequences=mock_sequences)
        mock_trainer.current_sampling_client.sample_async = AsyncMock(return_value=mock_result)

        request_data = {
            "prompt": "Hello world",
            "max_tokens": 50,
            "temperature": 1.0,
            "n": 2,
        }

        response = client.post("/v1/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["choices"]) == 2
        for i, choice in enumerate(data["choices"]):
            assert choice["index"] == i

    def test_batch_completions(self, client, mock_trainer):
        """Test batch completion request with list of prompts"""
        request_data = {
            "prompt": ["First prompt", "Second prompt"],
            "max_tokens": 50,
            "temperature": 0.8,
            "n": 1,
        }

        response = client.post("/v1/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["choices"]) == 2

    def test_with_stop_sequences(self, client, mock_trainer):
        """Test completion with stop sequences"""
        request_data = {
            "prompt": "Count to five:",
            "max_tokens": 100,
            "temperature": 0.7,
            "n": 1,
            "stop": ["\n", "five"],
        }

        response = client.post("/v1/completions", json=request_data)
        assert response.status_code == 200

        call_args = mock_trainer.current_sampling_client.sample_async.call_args
        sampling_params = call_args.kwargs["sampling_params"]
        assert sampling_params.stop == ["\n", "five"]


class TestChatCompletionsEndpoint:
    def test_chat_completion(self, client, mock_trainer):
        """Test basic chat completion"""
        request_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "n": 1,
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
        assert data["model"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert len(data["choices"]) == 1

        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Test output"
        assert choice["finish_reason"] == "stop"

    def test_chat_completion_multiple_samples(self, client, mock_trainer):
        mock_sequences = [
            SampledSequence(tokens=[10, 11], logprobs=[0.9, 0.8], stop_reason="stop"),
            SampledSequence(tokens=[12, 13], logprobs=[0.7, 0.6], stop_reason="stop"),
            SampledSequence(tokens=[14, 15], logprobs=[0.5, 0.4], stop_reason="stop"),
        ]
        mock_result = SampleResponse(sequences=mock_sequences)
        mock_trainer.current_sampling_client.sample_async = AsyncMock(return_value=mock_result)

        request_data = {
            "messages": [
                {"role": "user", "content": "Tell me three jokes"},
            ],
            "max_tokens": 200,
            "temperature": 1.0,
            "n": 3,
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["choices"]) == 3

    def test_chat_completion_with_stop(self, client, mock_trainer):
        request_data = {
            "messages": [
                {"role": "user", "content": "Count to 10"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "n": 1,
            "stop": ["10", "\n\n"],
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        call_args = mock_trainer.current_sampling_client.sample_async.call_args
        sampling_params = call_args.kwargs["sampling_params"]
        assert sampling_params.stop == ["10", "\n\n"]


class TestGenerateEndpoint:
    def test_generate_single_sample(self, client, mock_trainer):
        request_data = {
            "input_ids": [1, 2, 3, 4, 5],
            "sampling_params": {
                "n": 1,
                "max_new_tokens": 100,
                "temperature": 0.7,
                "stop": [],
            },
        }

        response = client.post("/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "text" in data
        assert "meta_info" in data
        assert data["text"] == "Test output"

        meta_info = data["meta_info"]
        assert meta_info["prompt_tokens"] == 5
        assert meta_info["completion_tokens"] == 3
        assert meta_info["finish_reason"] == "stop"
        assert "output_token_logprobs" in meta_info
        assert len(meta_info["output_token_logprobs"]) == 3

    def test_generate_multiple_samples(self, client, mock_trainer):
        mock_sequences = [
            SampledSequence(tokens=[10, 11], logprobs=[0.9, 0.8], stop_reason="stop"),
            SampledSequence(tokens=[12, 13], logprobs=[0.7, 0.6], stop_reason="stop"),
        ]
        mock_result = SampleResponse(sequences=mock_sequences)
        mock_trainer.current_sampling_client.sample_async = AsyncMock(return_value=mock_result)

        request_data = {
            "input_ids": [1, 2, 3],
            "sampling_params": {
                "n": 2,
                "max_new_tokens": 50,
                "temperature": 1.0,
            },
        }

        response = client.post("/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

        for result in data:
            assert "text" in result
            assert "meta_info" in result
            assert result["meta_info"]["prompt_tokens"] == 3

    def test_generate_without_input_ids(self, client):
        request_data = {
            "sampling_params": {
                "max_new_tokens": 100,
            },
        }

        response = client.post("/generate", json=request_data)
        assert response.status_code in [400, 500]
        assert "input_ids" in response.json()["detail"].lower()

    def test_generate_logprobs_format(self, client, mock_trainer):
        """Test that logprobs are formatted correctly for SGLang"""
        mock_sequence = SampledSequence(
            tokens=[100, 101, 102],
            logprobs=[-0.5, -0.3, -0.7],
            stop_reason="stop",
        )
        mock_result = SampleResponse(sequences=[mock_sequence])
        mock_trainer.current_sampling_client.sample_async = AsyncMock(return_value=mock_result)

        def decode_side_effect(tokens, **kwargs):
            if len(tokens) == 1:
                return f"token_{tokens[0]}"
            return "Full output"

        mock_trainer.tokenizer.decode = MagicMock(side_effect=decode_side_effect)

        request_data = {
            "input_ids": [1, 2],
            "sampling_params": {
                "n": 1,
                "max_new_tokens": 50,
            },
        }

        response = client.post("/generate", json=request_data)
        assert response.status_code == 200

        data = response.json()
        logprobs = data["meta_info"]["output_token_logprobs"]

        assert len(logprobs) == 3
        assert logprobs[0][0] == -0.5
        assert logprobs[0][1] == 100
        assert logprobs[0][2] == "token_100"


class TestErrorHandling:
    def test_completions_without_trainer(self, app_with_trainer):
        import tinker_atropos.trainer as trainer_module

        original_trainer = trainer_module.trainer
        trainer_module.trainer = None

        try:
            client = TestClient(app_with_trainer)
            response = client.post(
                "/v1/completions", json={"prompt": "test", "max_tokens": 10, "n": 1}
            )
            assert response.status_code == 503
            assert "Trainer not initialized" in response.json()["detail"]
        finally:
            trainer_module.trainer = original_trainer

    def test_chat_completions_without_trainer(self, app_with_trainer):
        import tinker_atropos.trainer as trainer_module

        original_trainer = trainer_module.trainer
        trainer_module.trainer = None

        try:
            client = TestClient(app_with_trainer)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10,
                    "n": 1,
                },
            )
            assert response.status_code == 503
            assert "Trainer not initialized" in response.json()["detail"]
        finally:
            trainer_module.trainer = original_trainer

    def test_generate_without_trainer(self, app_with_trainer):
        import tinker_atropos.trainer as trainer_module

        original_trainer = trainer_module.trainer
        trainer_module.trainer = None

        try:
            client = TestClient(app_with_trainer)
            response = client.post("/generate", json={"input_ids": [1, 2, 3]})
            assert response.status_code == 503
            assert "Trainer not initialized" in response.json()["detail"]
        finally:
            trainer_module.trainer = original_trainer
