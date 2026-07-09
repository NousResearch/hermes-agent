import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import twelvelabs

RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class TestTwelveLabsService(unittest.TestCase):
    """
    TwelveLabs 集成是完全 opt-in 的：未配置 twelvelabs_api_keys 时所有函数
    都必须是无副作用的 no-op，行为与不接入 TwelveLabs 完全一致。
    这些用例全部用 mock 替换 SDK 客户端，CI 不依赖真实网络或真实 API key。
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        twelvelabs._embed_text_cached.cache_clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        twelvelabs._embed_text_cached.cache_clear()

    # ---------------- disabled / no-op behavior ----------------

    def test_disabled_when_no_api_key(self):
        config.app.pop("twelvelabs_api_keys", None)
        self.assertFalse(twelvelabs.is_enabled())
        # rerank must return the input list unchanged
        terms = ["b", "a", "c"]
        self.assertEqual(
            twelvelabs.rerank_terms_by_subject("subject", terms), terms
        )
        # analyze must be a no-op returning None
        self.assertIsNone(twelvelabs.analyze_clip("https://x/y.mp4"))

    def test_rerank_skipped_when_flag_off(self):
        config.app["twelvelabs_api_keys"] = ["tlk_test"]
        config.app["twelvelabs_rerank_terms"] = False
        terms = ["b", "a"]
        # Even enabled, with the flag off we must not touch order or call the API.
        with patch.object(twelvelabs, "_client") as client:
            result = twelvelabs.rerank_terms_by_subject("subject", terms)
        self.assertEqual(result, terms)
        client.assert_not_called()

    # ---------------- enabled rerank behavior ----------------

    def _client_returning(self, vectors_by_text):
        """Build a fake TwelveLabs client whose embed.create returns canned vectors."""

        def fake_create(*, model_name, text):
            seg = MagicMock()
            seg.float_ = vectors_by_text[text]
            resp = MagicMock()
            resp.text_embedding.segments = [seg]
            return resp

        client = MagicMock()
        client.embed.create.side_effect = fake_create
        return client

    def test_rerank_orders_by_cosine_to_subject(self):
        config.app["twelvelabs_api_keys"] = ["tlk_test"]
        config.app["twelvelabs_rerank_terms"] = True

        # subject aligned with "city"; "kitten" is orthogonal.
        vectors = {
            "city skyline": [1.0, 0.0, 0.0],
            "downtown buildings": [0.9, 0.1, 0.0],  # close to subject
            "cute kitten": [0.0, 1.0, 0.0],  # far from subject
        }
        client = self._client_returning(vectors)

        with patch.object(twelvelabs, "_client", return_value=client):
            result = twelvelabs.rerank_terms_by_subject(
                "city skyline", ["cute kitten", "downtown buildings"]
            )

        # most relevant term must come first
        self.assertEqual(result, ["downtown buildings", "cute kitten"])

    def test_rerank_falls_back_on_embed_failure(self):
        config.app["twelvelabs_api_keys"] = ["tlk_test"]
        config.app["twelvelabs_rerank_terms"] = True

        client = MagicMock()
        client.embed.create.side_effect = RuntimeError("api down")

        terms = ["alpha", "beta"]
        with patch.object(twelvelabs, "_client", return_value=client):
            result = twelvelabs.rerank_terms_by_subject("subject", terms)

        # any failure must preserve the original order (never make things worse)
        self.assertEqual(result, terms)

    def test_rerank_noop_for_single_term(self):
        config.app["twelvelabs_api_keys"] = ["tlk_test"]
        config.app["twelvelabs_rerank_terms"] = True
        with patch.object(twelvelabs, "_client") as client:
            result = twelvelabs.rerank_terms_by_subject("subject", ["only"])
        self.assertEqual(result, ["only"])
        client.assert_not_called()

    # ---------------- analyze_clip ----------------

    def test_analyze_clip_returns_model_text(self):
        config.app["twelvelabs_api_keys"] = ["tlk_test"]

        # analyze_clip() lazily imports `twelvelabs.types.VideoContext_Url`.
        # The SDK is an optional extra, so the deterministic unit test must pass
        # even without `uv sync --extra twelvelabs`. Inject lightweight stub
        # modules so the internal import resolves; the mocked _client below does
        # the rest. (When the real SDK *is* installed, these stubs are ignored.)
        stub_types = type(sys)("twelvelabs.types")
        stub_types.VideoContext_Url = lambda *, url: {"url": url}
        stub_pkg = sys.modules.get("twelvelabs") or type(sys)("twelvelabs")
        with patch.dict(
            sys.modules, {"twelvelabs": stub_pkg, "twelvelabs.types": stub_types}
        ):
            self._run_analyze_clip_assertions()

    def _run_analyze_clip_assertions(self):
        resp = MagicMock()
        resp.data = "A city skyline at dusk."
        client = MagicMock()
        client.analyze.return_value = resp

        with patch.object(twelvelabs, "_client", return_value=client):
            out = twelvelabs.analyze_clip(
                "https://example.com/clip.mp4", prompt="describe"
            )

        self.assertEqual(out, "A city skyline at dusk.")
        # max_tokens must be clamped to the Pegasus minimum (>=512)
        self.assertGreaterEqual(client.analyze.call_args.kwargs["max_tokens"], 512)


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS and os.getenv("TWELVELABS_API_KEY"),
    "live test: set MPT_RUN_INTEGRATION_TESTS=1 and TWELVELABS_API_KEY to run "
    "against the real TwelveLabs API",
)
class TestTwelveLabsLive(unittest.TestCase):
    """Live contract check — only runs with MPT_RUN_INTEGRATION_TESTS=1 + a key."""

    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["twelvelabs_api_keys"] = [os.environ["TWELVELABS_API_KEY"]]
        config.app["twelvelabs_rerank_terms"] = True
        twelvelabs._embed_text_cached.cache_clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        twelvelabs._embed_text_cached.cache_clear()

    def test_marengo_embedding_is_512_dim(self):
        vec = twelvelabs.embed_text("a city skyline at night")
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), 512)

    def test_rerank_puts_relevant_term_first(self):
        result = twelvelabs.rerank_terms_by_subject(
            "city skyline at night",
            ["cute kitten playing with yarn", "downtown buildings and traffic at dusk"],
        )
        self.assertEqual(result[0], "downtown buildings and traffic at dusk")


if __name__ == "__main__":
    unittest.main()
