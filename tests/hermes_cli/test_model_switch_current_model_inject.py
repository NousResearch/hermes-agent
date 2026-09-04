"""Current-model injection must fold picker-search aliases before membership.

Kimi Coding Plan's flagship is configured/discovered as the bare wire id
``k3`` while the curated picker catalog carries the public slug ``kimi-k3``
(the merge dedups ``k3`` into it). The injection post-pass that prepends the
active model to its provider row must treat them as the same model — an
exact-string check re-injected the bare id and rendered one model as two
picker rows ("K3" + "Kimi K3").
"""

from hermes_cli.model_switch import _inject_current_model_row


def _rows(models, *, is_current=True):
    return [
        {
            "slug": "kimi-coding",
            "is_current": is_current,
            "models": list(models),
            "total_models": len(models),
        }
    ]


class TestAliasAwareMembership:
    def test_bare_k3_not_injected_when_curated_slug_present(self):
        rows = _rows(["kimi-k3", "kimi-k2.7-code", "k3-256k"])
        _inject_current_model_row(rows, "k3")
        assert rows[0]["models"] == ["kimi-k3", "kimi-k2.7-code", "k3-256k"]
        assert rows[0]["total_models"] == 3

    def test_curated_slug_not_injected_when_bare_id_present(self):
        rows = _rows(["k3", "kimi-k2.7-code"])
        _inject_current_model_row(rows, "kimi-k3")
        assert rows[0]["models"] == ["k3", "kimi-k2.7-code"]

    def test_case_insensitive_match(self):
        rows = _rows(["Kimi-K3"])
        _inject_current_model_row(rows, "k3")
        assert rows[0]["models"] == ["Kimi-K3"]


class TestGenuineInjectionStillWorks:
    def test_uncurated_model_is_prepended(self):
        rows = _rows(["kimi-k3", "kimi-k2.7-code"])
        _inject_current_model_row(rows, "my-custom-kimi")
        assert rows[0]["models"] == ["my-custom-kimi", "kimi-k3", "kimi-k2.7-code"]
        assert rows[0]["total_models"] == 3

    def test_k3_256k_is_not_folded_into_k3(self):
        # k3-256k is a distinct live id, not an alias of k3.
        rows = _rows(["kimi-k3"])
        _inject_current_model_row(rows, "k3-256k")
        assert rows[0]["models"] == ["k3-256k", "kimi-k3"]

    def test_non_current_row_untouched(self):
        rows = _rows(["kimi-k3"], is_current=False)
        _inject_current_model_row(rows, "my-custom-kimi")
        assert rows[0]["models"] == ["kimi-k3"]

    def test_empty_current_model_noop(self):
        rows = _rows(["kimi-k3"])
        _inject_current_model_row(rows, "")
        assert rows[0]["models"] == ["kimi-k3"]

    def test_empty_row_models_still_injects(self):
        rows = _rows([])
        _inject_current_model_row(rows, "k3")
        assert rows[0]["models"] == ["k3"]


class TestNativeCatalogEmptyGuard:
    def test_empty_native_catalog_row_not_injected(self):
        # main's native-discovery parity guard (fa1bb88e / #81e81350): a row
        # whose live catalog is deliberately empty must not receive an
        # injected uncurated model.
        rows = _rows([])
        rows[0]["native_catalog_empty"] = True
        _inject_current_model_row(rows, "my-custom-kimi")
        assert rows[0]["models"] == []
        assert rows[0]["total_models"] == 0

    def test_guard_does_not_block_other_current_rows(self):
        guarded = _rows(["placeholder"])[0]
        guarded["native_catalog_empty"] = True
        guarded["slug"] = "guarded-provider"
        guarded["models"] = []
        guarded["total_models"] = 0
        normal = _rows(["kimi-k3"])[0]
        rows = [guarded, normal]
        _inject_current_model_row(rows, "my-custom-kimi")
        assert guarded["models"] == []
        assert normal["models"] == ["my-custom-kimi", "kimi-k3"]
