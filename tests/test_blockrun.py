"""
End-to-end test suite for the BlockRun / ClawRouter hermes-agent integration.

Test groups:
  Unit   — no network, no real wallet needed (mocked or hardhat keys)
  Live   — requires BLOCKRUN_WALLET_KEY + funded wallet (marked with @pytest.mark.live)

Run unit tests only (default):
  cd hermes-blockrun-integration && pytest tests/ -v

Run all including live API tests:
  pytest tests/ -v -m live
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# Make the integration root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Hardhat/Anvil well-known test key (never has real funds)
_TEST_ETH_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
_TEST_ETH_ADDR = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"


def _fresh_solana_key() -> str:
    """Generate a real Solana keypair for testing (no funds, safe)."""
    from blockrun_llm.solana_wallet import create_solana_wallet
    return create_solana_wallet()["private_key"]


# ── Stub hermes registry so blockrun_tool.py can be imported standalone ──────
_ROOT = os.path.join(os.path.dirname(__file__), "..")

class _FakeRegistry:
    def register(self, **_):
        pass

# Inject a fake tools.registry before anything tries to import blockrun_tool
_fake_registry_module = type(sys)("tools.registry")
_fake_registry_module.registry = _FakeRegistry()
sys.modules["tools.registry"] = _fake_registry_module

# Make "tools" a real package backed by our tools/ directory
if "tools" not in sys.modules:
    import importlib
    import importlib.util
    _tools_init = os.path.join(_ROOT, "tools", "__init__.py")
    # Create __init__.py if missing (makes tools/ a proper package)
    if not os.path.exists(_tools_init):
        open(_tools_init, "w").close()
    spec = importlib.util.spec_from_file_location(
        "tools", _tools_init,
        submodule_search_locations=[os.path.join(_ROOT, "tools")]
    )
    _tools_pkg = importlib.util.module_from_spec(spec)
    sys.modules["tools"] = _tools_pkg
    spec.loader.exec_module(_tools_pkg)


# ─────────────────────────────────────────────────────────────────────────────
# Group 1: Imports
# ─────────────────────────────────────────────────────────────────────────────

class TestImports(unittest.TestCase):

    def test_provider_module_imports(self):
        from hermes_cli import blockrun_provider
        self.assertTrue(hasattr(blockrun_provider, "resolve_blockrun_provider"))
        self.assertTrue(hasattr(blockrun_provider, "BlockRunX402Transport"))
        self.assertTrue(hasattr(blockrun_provider, "AsyncBlockRunX402Transport"))
        self.assertTrue(hasattr(blockrun_provider, "create_sync_client"))
        self.assertTrue(hasattr(blockrun_provider, "create_async_client"))

    def test_tool_module_imports(self):
        from tools import blockrun_tool  # noqa: F401 — side-effects register tools


# ─────────────────────────────────────────────────────────────────────────────
# Group 2: Chain detection
# ─────────────────────────────────────────────────────────────────────────────

class TestChainDetection(unittest.TestCase):

    def _detect(self, env: dict) -> str:
        from hermes_cli.blockrun_provider import _detect_chain, _BASE_KEY_VARS, _SOLANA_KEY_VARS
        # Temporarily patch only the relevant env vars
        all_vars = list(_BASE_KEY_VARS) + list(_SOLANA_KEY_VARS) + ["BLOCKRUN_CHAIN"]
        clean = {v: "" for v in all_vars}
        clean.update(env)
        with patch.dict(os.environ, clean, clear=False):
            # Remove the vars we want absent
            for k, v in clean.items():
                if v == "":
                    os.environ.pop(k, None)
            return _detect_chain()

    def test_default_is_base(self):
        result = self._detect({})
        self.assertEqual(result, "base")

    def test_blockrun_chain_env_forces_solana(self):
        result = self._detect({"BLOCKRUN_CHAIN": "solana"})
        self.assertEqual(result, "solana")

    def test_only_solana_key_picks_solana(self):
        result = self._detect({"SOLANA_WALLET_KEY": "somekey"})
        self.assertEqual(result, "solana")

    def test_both_keys_defaults_to_base(self):
        result = self._detect({
            "BLOCKRUN_WALLET_KEY": "0xabc",
            "SOLANA_WALLET_KEY": "somekey",
        })
        self.assertEqual(result, "base")

    def test_session_file_does_not_affect_chain_detection(self):
        """~/.blockrun/.session must NOT influence chain detection."""
        # Even if the session file exists (it does on this machine),
        # chain detection should only look at env vars.
        result = self._detect({"SOLANA_WALLET_KEY": "somekey"})
        self.assertEqual(result, "solana")


# ─────────────────────────────────────────────────────────────────────────────
# Group 3: Wallet key loading
# ─────────────────────────────────────────────────────────────────────────────

class TestWalletKeyLoading(unittest.TestCase):

    def test_base_key_from_env(self):
        from hermes_cli.blockrun_provider import _load_base_key
        with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": "0xdeadbeef"}):
            self.assertEqual(_load_base_key(), "0xdeadbeef")

    def test_base_legacy_key_var(self):
        from hermes_cli.blockrun_provider import _load_base_key
        env = {"BLOCKRUN_WALLET_KEY": "", "BASE_CHAIN_WALLET_KEY": "0xlegacy"}
        with patch.dict(os.environ, env):
            os.environ.pop("BLOCKRUN_WALLET_KEY", None)
            self.assertEqual(_load_base_key(), "0xlegacy")

    def test_solana_key_from_env(self):
        from hermes_cli.blockrun_provider import _load_solana_key
        with patch.dict(os.environ, {"SOLANA_WALLET_KEY": "solanakey123"}):
            self.assertEqual(_load_solana_key(), "solanakey123")

    def test_base_key_falls_back_to_session_file(self):
        from hermes_cli.blockrun_provider import _load_base_key
        session = os.path.expanduser("~/.blockrun/.session")
        # This machine has a session file — verify it's read
        if os.path.exists(session):
            key = _load_base_key()
            self.assertIsNotNone(key)
            self.assertTrue(len(key) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# Group 4: Signer loading
# ─────────────────────────────────────────────────────────────────────────────

class TestSignerLoading(unittest.TestCase):

    def test_base_signer_returns_local_account(self):
        from hermes_cli.blockrun_provider import _load_signer
        from eth_account import Account
        signer = _load_signer(_TEST_ETH_KEY, "base")
        self.assertIsInstance(signer, Account.from_key(_TEST_ETH_KEY).__class__)
        self.assertEqual(signer.address.lower(), _TEST_ETH_ADDR.lower())

    def test_solana_signer_returns_x402_client(self):
        from hermes_cli.blockrun_provider import _load_signer
        from blockrun_llm.solana_client import x402ClientSync
        signer = _load_signer(_fresh_solana_key(), "solana")
        self.assertIsInstance(signer, x402ClientSync)

    def test_invalid_base_key_raises(self):
        from hermes_cli.blockrun_provider import _load_signer
        with self.assertRaises(Exception):
            _load_signer("not-a-valid-key", "base")


# ─────────────────────────────────────────────────────────────────────────────
# Group 5: Transport instantiation
# ─────────────────────────────────────────────────────────────────────────────

class TestTransportInstantiation(unittest.TestCase):

    def test_sync_base_transport(self):
        from hermes_cli.blockrun_provider import BlockRunX402Transport
        t = BlockRunX402Transport(private_key=_TEST_ETH_KEY, chain="base")
        self.assertEqual(t._chain, "base")

    def test_async_base_transport(self):
        from hermes_cli.blockrun_provider import AsyncBlockRunX402Transport
        t = AsyncBlockRunX402Transport(private_key=_TEST_ETH_KEY, chain="base")
        self.assertEqual(t._chain, "base")

    def test_sync_solana_transport(self):
        from hermes_cli.blockrun_provider import BlockRunX402Transport
        from blockrun_llm.solana_client import x402ClientSync
        t = BlockRunX402Transport(private_key=_fresh_solana_key(), chain="solana")
        self.assertEqual(t._chain, "solana")
        self.assertIsInstance(t._signer, x402ClientSync)

    def test_transport_passes_through_200(self):
        """Transport must not touch non-402 responses."""
        from hermes_cli.blockrun_provider import BlockRunX402Transport
        import httpx

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.headers = {}

        fake_req = httpx.Request("GET", "https://blockrun.ai/api/v1/chat/completions")
        t = BlockRunX402Transport(private_key=_TEST_ETH_KEY, chain="base")
        with patch.object(httpx.HTTPTransport, "handle_request", return_value=mock_200):
            result = t.handle_request(fake_req)
        self.assertEqual(result.status_code, 200)

    def test_transport_passes_through_402_without_payment_header(self):
        """402 with no X-Payment-Required header must pass through unchanged."""
        from hermes_cli.blockrun_provider import BlockRunX402Transport
        import httpx

        mock_402 = MagicMock()
        mock_402.status_code = 402
        mock_402.headers = {}

        fake_req = httpx.Request("GET", "https://blockrun.ai/api/v1/chat/completions")
        t = BlockRunX402Transport(private_key=_TEST_ETH_KEY, chain="base")
        with patch.object(httpx.HTTPTransport, "handle_request", return_value=mock_402):
            result = t.handle_request(fake_req)
        self.assertEqual(result.status_code, 402)


# ─────────────────────────────────────────────────────────────────────────────
# Group 6: Client factory
# ─────────────────────────────────────────────────────────────────────────────

class TestClientFactory(unittest.TestCase):

    def test_create_sync_client_base(self):
        import openai
        from hermes_cli.blockrun_provider import create_sync_client, BLOCKRUN_BASE_URL
        client = create_sync_client(_TEST_ETH_KEY, chain="base")
        self.assertIsInstance(client, openai.OpenAI)
        self.assertIn("blockrun.ai", str(client.base_url))

    def test_create_sync_client_solana(self):
        import openai
        from hermes_cli.blockrun_provider import create_sync_client
        client = create_sync_client(_fresh_solana_key(), chain="solana")
        self.assertIsInstance(client, openai.OpenAI)
        self.assertIn("sol.blockrun.ai", str(client.base_url))

    def test_create_async_client_base(self):
        import openai
        from hermes_cli.blockrun_provider import create_async_client
        client = create_async_client(_TEST_ETH_KEY, chain="base")
        self.assertIsInstance(client, openai.AsyncOpenAI)


# ─────────────────────────────────────────────────────────────────────────────
# Group 7: Provider resolver
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderResolver(unittest.TestCase):

    def test_resolve_base(self):
        from hermes_cli.blockrun_provider import resolve_blockrun_provider, BLOCKRUN_BASE_URL
        result = resolve_blockrun_provider(
            explicit_api_key=_TEST_ETH_KEY,
            explicit_base_url=None,
            chain="base",
        )
        self.assertEqual(result["provider"], "blockrun")
        self.assertEqual(result["api_mode"], "chat_completions")
        self.assertEqual(result["base_url"], BLOCKRUN_BASE_URL)
        self.assertEqual(result["blockrun_chain"], "base")
        self.assertIn("blockrun_client_factory", result)
        self.assertIn("blockrun_async_client_factory", result)

    def test_resolve_solana(self):
        from hermes_cli.blockrun_provider import resolve_blockrun_provider, BLOCKRUN_SOLANA_URL
        result = resolve_blockrun_provider(
            explicit_api_key=_fresh_solana_key(),
            explicit_base_url=None,
            chain="solana",
        )
        self.assertEqual(result["blockrun_chain"], "solana")
        self.assertEqual(result["base_url"], BLOCKRUN_SOLANA_URL)

    def test_resolve_testnet(self):
        from hermes_cli.blockrun_provider import resolve_blockrun_provider, BLOCKRUN_BASE_TESTNET
        with patch.dict(os.environ, {"NETWORK_MODE": "testnet"}):
            result = resolve_blockrun_provider(
                explicit_api_key=_TEST_ETH_KEY,
                explicit_base_url=None,
                chain="base",
            )
        self.assertEqual(result["base_url"], BLOCKRUN_BASE_TESTNET)

    def test_resolve_explicit_base_url_wins(self):
        from hermes_cli.blockrun_provider import resolve_blockrun_provider
        result = resolve_blockrun_provider(
            explicit_api_key=_TEST_ETH_KEY,
            explicit_base_url="https://custom.blockrun.ai/api/v1",
            chain="base",
        )
        self.assertEqual(result["base_url"], "https://custom.blockrun.ai/api/v1")

    def test_resolve_no_key_raises_value_error(self):
        from hermes_cli.blockrun_provider import resolve_blockrun_provider
        env = {k: "" for k in ("BLOCKRUN_WALLET_KEY", "BASE_CHAIN_WALLET_KEY")}
        with patch.dict(os.environ, env):
            for k in env:
                os.environ.pop(k, None)
            # Also ensure no session file interferes — mock _load_base_key
            with patch("hermes_cli.blockrun_provider._load_base_key", return_value=None):
                with self.assertRaises(ValueError) as ctx:
                    resolve_blockrun_provider(
                        explicit_api_key=None,
                        explicit_base_url=None,
                        chain="base",
                    )
        self.assertIn("BLOCKRUN_WALLET_KEY", str(ctx.exception))

    def test_client_factory_callable(self):
        from hermes_cli.blockrun_provider import resolve_blockrun_provider
        import openai
        result = resolve_blockrun_provider(
            explicit_api_key=_TEST_ETH_KEY,
            explicit_base_url=None,
            chain="base",
        )
        client = result["blockrun_client_factory"]()
        self.assertIsInstance(client, openai.OpenAI)


# ─────────────────────────────────────────────────────────────────────────────
# Group 8: Tool schemas
# ─────────────────────────────────────────────────────────────────────────────

class TestToolSchemas(unittest.TestCase):

    def setUp(self):
        from tools import blockrun_tool  # ensures all tools are registered
        self.tool = blockrun_tool

    def _schema(self, name):
        schemas = {
            "blockrun_wallet_setup":          self.tool.WALLET_SETUP_SCHEMA,
            "blockrun_wallet_balance":        self.tool.WALLET_BALANCE_SCHEMA,
            "blockrun_wallet_address":        self.tool.WALLET_ADDRESS_SCHEMA,
            "blockrun_solana_wallet_setup":   self.tool.SOLANA_WALLET_SETUP_SCHEMA,
            "blockrun_solana_wallet_balance": self.tool.SOLANA_WALLET_BALANCE_SCHEMA,
            "blockrun_image_generate":        self.tool.IMAGE_GENERATE_SCHEMA,
            "blockrun_image_edit":            self.tool.IMAGE_EDIT_SCHEMA,
            "blockrun_prediction_markets":    self.tool.PREDICTION_MARKETS_SCHEMA,
        }
        return schemas[name]

    def test_all_schemas_have_required_fields(self):
        names = [
            "blockrun_wallet_setup", "blockrun_wallet_balance",
            "blockrun_wallet_address", "blockrun_solana_wallet_setup",
            "blockrun_solana_wallet_balance", "blockrun_image_generate",
            "blockrun_image_edit", "blockrun_prediction_markets",
        ]
        for name in names:
            schema = self._schema(name)
            self.assertIn("name", schema, f"{name}: missing 'name'")
            self.assertIn("description", schema, f"{name}: missing 'description'")
            self.assertIn("parameters", schema, f"{name}: missing 'parameters'")
            self.assertEqual(schema["name"], name, f"{name}: name mismatch")

    def test_image_generate_model_enum(self):
        schema = self._schema("blockrun_image_generate")
        model_prop = schema["parameters"]["properties"]["model"]
        self.assertIn("enum", model_prop)
        self.assertIn("nano-banana", model_prop["enum"])
        self.assertIn("dall-e-3", model_prop["enum"])

    def test_image_edit_requires_prompt_and_image(self):
        schema = self._schema("blockrun_image_edit")
        self.assertIn("prompt", schema["parameters"]["required"])
        self.assertIn("image", schema["parameters"]["required"])


# ─────────────────────────────────────────────────────────────────────────────
# Group 9: Tool handlers — error cases (no real API)
# ─────────────────────────────────────────────────────────────────────────────

class TestToolHandlerErrors(unittest.TestCase):

    def _clear_wallet_env(self):
        for k in ("BLOCKRUN_WALLET_KEY", "BASE_CHAIN_WALLET_KEY", "SOLANA_WALLET_KEY"):
            os.environ.pop(k, None)

    def test_wallet_balance_no_key(self):
        from tools.blockrun_tool import _handle_wallet_balance
        self._clear_wallet_env()
        with patch("hermes_cli.blockrun_provider._load_base_key", return_value=None):
            result = _handle_wallet_balance({})
        self.assertIn("❌", result)

    def test_image_generate_no_prompt(self):
        from tools.blockrun_tool import _handle_image_generate
        result = _handle_image_generate({})
        self.assertIn("❌", result)
        self.assertIn("prompt", result)

    def test_image_edit_no_prompt(self):
        from tools.blockrun_tool import _handle_image_edit
        result = _handle_image_edit({"image": "data:image/png;base64,abc"})
        self.assertIn("❌", result)
        self.assertIn("prompt", result)

    def test_image_edit_no_image(self):
        from tools.blockrun_tool import _handle_image_edit
        result = _handle_image_edit({"prompt": "make it blue"})
        self.assertIn("❌", result)
        self.assertIn("image", result)

    def test_prediction_markets_no_path_returns_help(self):
        from tools.blockrun_tool import _handle_prediction_markets
        result = _handle_prediction_markets({})
        self.assertIn("polymarket", result.lower())
        self.assertIn("kalshi", result.lower())

    def test_prediction_markets_no_key(self):
        from tools.blockrun_tool import _handle_prediction_markets
        self._clear_wallet_env()
        with patch("hermes_cli.blockrun_provider._load_base_key", return_value=None):
            result = _handle_prediction_markets({"path": "polymarket/markets"})
        self.assertIn("❌", result)


# ─────────────────────────────────────────────────────────────────────────────
# Group 10: Live API tests (require funded wallet)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestLiveAPI(unittest.TestCase):
    """
    These tests hit the real BlockRun API and consume small amounts of USDC.
    Run with: pytest tests/ -v -m live
    Requires: BLOCKRUN_WALLET_KEY set and wallet funded with USDC on Base.
    """

    def setUp(self):
        self.key = os.environ.get("BLOCKRUN_WALLET_KEY")
        if not self.key:
            # Try session file
            session = os.path.expanduser("~/.blockrun/.session")
            if os.path.exists(session):
                self.key = open(session).read().strip()
        if not self.key:
            self.skipTest("BLOCKRUN_WALLET_KEY not set and no ~/.blockrun/.session")

    def test_wallet_balance_live(self):
        from blockrun_llm import LLMClient
        client = LLMClient(private_key=self.key)
        balance = client.get_balance()
        self.assertIsInstance(balance, float)
        self.assertGreaterEqual(balance, 0)
        print(f"\n  Balance: ${balance:.4f} USDC")

    def test_llm_completion_live(self):
        """Basic LLM call through BlockRun x402 (uses cheap NVIDIA model)."""
        from hermes_cli.blockrun_provider import create_sync_client
        client = create_sync_client(self.key, chain="base")
        resp = client.chat.completions.create(
            model="nvidia/gpt-oss-20b",  # smallest / cheapest
            messages=[{"role": "user", "content": "Reply with just the word: PASS"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        print(f"\n  LLM response: {text!r}")
        self.assertIsNotNone(text)
        self.assertTrue(len(text) > 0)

    def test_image_generate_live(self):
        """Image generation via x402 (uses cheapest model)."""
        from tools.blockrun_tool import _handle_image_generate
        with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": self.key}):
            result = _handle_image_generate({
                "prompt": "a simple blue circle on white background",
                "model": "gpt-image-1",
                "size": "1024x1024",
                "n": 1,
            })
        print(f"\n  Image result: {result[:200]}")
        self.assertIn("✅", result)
        # GPT Image 1 returns base64 data URIs; other models return https URLs
        self.assertTrue(
            "https://" in result or "data:image/" in result,
            f"Expected image URL or data URI in result: {result[:300]}"
        )

    def test_prediction_markets_live(self):
        """Prediction market data query (Polymarket markets list)."""
        from tools.blockrun_tool import _handle_prediction_markets
        with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": self.key}):
            result = _handle_prediction_markets({"path": "polymarket/markets"})
        print(f"\n  PM result: {result[:300]}")
        self.assertNotIn("❌", result)


# ─────────────────────────────────────────────────────────────────────────────
# Group 11: Multi-model live tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.live
class TestLiveMultiModel(unittest.TestCase):
    """
    Tests that exercise each supported model family through the x402 transport.
    Uses the cheapest model per provider to minimise cost (~$0.001 per test).

    Run with: pytest tests/ -v -m live
    """

    # cheapest confirmed-working model per family
    MODELS = [
        ("nvidia",    "nvidia/gpt-oss-20b",           "free NVIDIA baseline"),
        ("openai",    "openai/gpt-5-mini",             "OpenAI GPT-5 mini"),
        ("anthropic", "anthropic/claude-haiku-4.5",   "Anthropic Claude Haiku"),
        ("google",    "google/gemini-2.5-flash",      "Google Gemini Flash"),
    ]

    PROMPT = "Reply with exactly one word: PASS"

    def setUp(self):
        key = os.environ.get("BLOCKRUN_WALLET_KEY")
        if not key:
            session = os.path.expanduser("~/.blockrun/.session")
            if os.path.exists(session):
                key = open(session).read().strip()
        if not key:
            self.skipTest("BLOCKRUN_WALLET_KEY not set")
        from hermes_cli.blockrun_provider import create_sync_client
        self.client = create_sync_client(key, chain="base")

    def _chat(self, model: str, messages: list, **kwargs) -> str:
        kwargs.setdefault("max_tokens", 20)
        resp = self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content.strip()

    # ── one test per model family ──────────────────────────────────────────

    def test_nvidia_basic(self):
        text = self._chat("nvidia/gpt-oss-20b",
                          [{"role": "user", "content": self.PROMPT}])
        print(f"\n  nvidia: {text!r}")
        self.assertTrue(len(text) > 0)

    def test_openai_basic(self):
        text = self._chat("openai/gpt-5-mini",
                          [{"role": "user", "content": self.PROMPT}])
        print(f"\n  openai: {text!r}")
        self.assertTrue(len(text) > 0)

    def test_anthropic_basic(self):
        text = self._chat("anthropic/claude-haiku-4.5",
                          [{"role": "user", "content": self.PROMPT}])
        print(f"\n  anthropic: {text!r}")
        self.assertTrue(len(text) > 0)

    def test_google_basic(self):
        text = self._chat("google/gemini-2.5-flash",
                          [{"role": "user", "content": self.PROMPT}])
        print(f"\n  google: {text!r}")
        self.assertTrue(len(text) > 0)

    # ── system prompt ──────────────────────────────────────────────────────

    def test_system_prompt(self):
        """System prompt must be respected across providers."""
        messages = [
            {"role": "system", "content": "You are a robot. Always end every reply with 'BEEP'."},
            {"role": "user",   "content": "Say hello."},
        ]
        text = self._chat("openai/gpt-5-mini", messages, max_tokens=30)
        print(f"\n  system prompt: {text!r}")
        self.assertIn("BEEP", text.upper())

    # ── multi-turn ─────────────────────────────────────────────────────────

    def test_multi_turn(self):
        """Conversation history must be maintained."""
        messages = [
            {"role": "user",      "content": "My secret number is 42. Remember it."},
            {"role": "assistant", "content": "Got it, your secret number is 42."},
            {"role": "user",      "content": "What is my secret number? Reply with just the number."},
        ]
        text = self._chat("openai/gpt-5-mini", messages, max_tokens=30)
        print(f"\n  multi-turn: {text!r}")
        self.assertIn("42", text)

    # ── streaming ──────────────────────────────────────────────────────────

    def test_streaming(self):
        """Streaming must return at least one chunk with content."""
        chunks = []
        stream = self.client.chat.completions.create(
            model="openai/gpt-5-mini",
            messages=[{"role": "user", "content": "Count to 3."}],
            max_tokens=20,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)
        text = "".join(chunks)
        print(f"\n  streaming: {text!r} ({len(chunks)} chunks)")
        self.assertTrue(len(chunks) > 0, "Expected at least one streaming chunk")
        self.assertTrue(len(text) > 0)

    # ── tool calling ───────────────────────────────────────────────────────

    def test_tool_calling(self):
        """Model must be able to invoke a tool (function calling)."""
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        }]
        resp = self.client.chat.completions.create(
            model="openai/gpt-5-mini",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=256,
        )
        msg = resp.choices[0].message
        print(f"\n  tool calling finish_reason: {resp.choices[0].finish_reason}")
        # Must either call the tool or give a text answer
        has_tool_call = bool(msg.tool_calls)
        has_content   = bool(msg.content and len(msg.content) > 0)
        self.assertTrue(
            has_tool_call or has_content,
            "Expected either a tool call or text content"
        )
        if has_tool_call:
            fn = msg.tool_calls[0].function
            print(f"  tool called: {fn.name}({fn.arguments})")
            self.assertEqual(fn.name, "get_weather")
            self.assertIn("Tokyo", fn.arguments)

    # ── model list ─────────────────────────────────────────────────────────

    def test_models_endpoint(self):
        """GET /models must return our known families."""
        models = self.client.models.list()
        ids = {m.id for m in models.data}
        print(f"\n  total models: {len(ids)}")
        for family, model_id, label in self.MODELS:
            self.assertIn(model_id, ids,
                          f"Expected {model_id} ({label}) in model list")

    # ── user-agent header ──────────────────────────────────────────────────

    def test_user_agent_header_sent(self):
        """Requests must carry the hermes-agent User-Agent header.

        The spy is placed on httpx.HTTPTransport.handle_request (the super)
        so it sees the request AFTER BlockRunX402Transport._inject_ua has
        stamped the hermes-agent User-Agent onto it.
        """
        import httpx as _httpx
        captured = {}

        original_super = _httpx.HTTPTransport.handle_request

        def spy(self_t, request):
            if "blockrun.ai" in str(request.url):
                captured["user_agent"] = request.headers.get("user-agent", "")
            return original_super(self_t, request)

        with patch.object(_httpx.HTTPTransport, "handle_request", spy):
            self._chat("nvidia/gpt-oss-20b",
                       [{"role": "user", "content": "hi"}])

        ua = captured.get("user_agent", "")
        print(f"\n  User-Agent: {ua!r}")
        self.assertIn("hermes-agent", ua.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
