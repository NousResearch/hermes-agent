"""Regression tests for MiniMax OAuth ``verification_uri`` host rewrite (#19337).

MiniMax's OAuth authorization endpoint currently returns
``https://www.minimaxi.io/oauth-authorize?…`` as ``verification_uri``. That path
307-redirects to ``/`` (the marketing homepage), so a user following the printed
URL cannot approve the device code. The live approval page lives at
``platform.minimaxi.io`` with the same path and query.

The rewrite is a defensive client-side normalization in
``hermes_cli.auth_minimax._minimax_rewrite_verification_uri``; it must:

* Rewrite ``www.minimaxi.io`` → ``platform.minimaxi.io`` when the path is
  ``/oauth-authorize[/*]``.
* Leave every other host or path untouched (incl. ``www.minimaxi.io`` paths
  that are NOT the OAuth approval page, and the legacy ``minimaxi.com`` host).
* Preserve query string, scheme, and port (when present).
"""

from __future__ import annotations

from hermes_cli.auth_minimax import _minimax_rewrite_verification_uri


class TestRewriteVerificationUri:
    def test_rewrites_www_to_platform_for_oauth_authorize(self):
        # The exact URL pattern the upstream OAuth endpoint currently returns.
        uri = "https://www.minimaxi.io/oauth-authorize?user_code=ABCD&client=OpenClaw"
        assert _minimax_rewrite_verification_uri(uri) == (
            "https://platform.minimaxi.io/oauth-authorize?user_code=ABCD&client=OpenClaw"
        )

    def test_rewrites_with_subpath(self):
        # Defensive: any path starting with /oauth-authorize should be rewritten.
        uri = "https://www.minimaxi.io/oauth-authorize/extra?x=1"
        assert _minimax_rewrite_verification_uri(uri).startswith("https://platform.minimaxi.io/oauth-authorize/extra?")

    def test_leaves_already_correct_host(self):
        # No-op once the upstream fixes its response.
        uri = "https://platform.minimaxi.io/oauth-authorize?user_code=ABCD"
        assert _minimax_rewrite_verification_uri(uri) == uri

    def test_leaves_other_paths_on_www(self):
        # The /oauth-authorize scope is the only thing we know about; other paths
        # on the same host may serve other content that the user does want.
        uri = "https://www.minimaxi.io/some/other/path?x=1"
        assert _minimax_rewrite_verification_uri(uri) == uri

    def test_leaves_legacy_minimaxi_com_host(self):
        # Older ``minimaxi.com`` (no www) host must NOT be rewritten — different
        # host entirely, the rewrite is www → platform only.
        uri = "https://minimaxi.com/oauth-authorize?x=1"
        assert _minimax_rewrite_verification_uri(uri) == uri

    def test_preserves_query_string_order(self):
        # Query is passed through untouched (urllib preserves order on urlunparse).
        uri = "https://www.minimaxi.io/oauth-authorize?b=2&a=1&c=3"
        rewritten = _minimax_rewrite_verification_uri(uri)
        assert "b=2&a=1&c=3" in rewritten
        assert rewritten.startswith("https://platform.minimaxi.io/oauth-authorize?")

    def test_no_op_for_unrelated_host(self):
        # Non-MiniMax hosts are never touched.
        uri = "https://example.com/oauth-authorize?x=1"
        assert _minimax_rewrite_verification_uri(uri) == uri

    def test_invalid_url_returns_input_unchanged(self):
        # urlparse may raise ValueError on truly malformed URIs; the helper
        # must swallow it and return the input verbatim (caller still prints it).
        uri = "not a url at all"
        assert _minimax_rewrite_verification_uri(uri) == uri