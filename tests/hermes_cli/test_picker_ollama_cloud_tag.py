"""The current-model injection in ``_finalize_picker_rows`` must not create a twin row when the
provider already lists the model under an equivalent spelling. Ollama endpoints accept the bare
id and its ``:cloud`` tag form interchangeably on completions while ``/v1/models`` lists the
bare form, so a pinned ``:cloud`` pick must fold onto the existing bare row (and vice versa);
non-ollama providers keep injecting the pick verbatim."""

from hermes_cli.model_switch_providers import _finalize_picker_rows, same_ollama_model


def _row(slug: str, models: list[str]) -> dict:
    return {
        "slug": slug,
        "provider_id": slug,
        "name": slug,
        "is_current": True,
        "is_user_defined": False,
        "models": list(models),
        "total_models": len(models),
        "source": "built-in",
    }


def test_ollama_cloud_pick_folds_to_existing_bare_row():
    rows = [_row("ollama-cloud", ["glm-5.3-flash", "glm-5.3", "beam"])]
    out = _finalize_picker_rows(rows, {}, "glm-5.3-flash:cloud")
    assert out[0]["models"] == ["glm-5.3-flash", "glm-5.3", "beam"]
    assert out[0]["total_models"] == 3


def test_reverse_bare_pick_folds_to_cloud_listed_row():
    rows = [_row("ollama", ["glm-5.3-flash:cloud"])]
    out = _finalize_picker_rows(rows, {}, "glm-5.3-flash")
    assert out[0]["models"] == ["glm-5.3-flash:cloud"]


def test_non_ollama_provider_still_injects_the_pick():
    rows = [_row("anthropic", ["claude-opus-4-6"])]
    out = _finalize_picker_rows(rows, {}, "claude-opus-4-6:cloud")
    assert out[0]["models"][0] == "claude-opus-4-6:cloud"
    assert out[0]["total_models"] == 2


def test_same_ollama_model_is_slug_gated_and_folds_only_the_cloud_tag():
    assert same_ollama_model("ollama-cloud", "glm-5.3-flash:cloud", "GLM-5.3-Flash")
    assert same_ollama_model("ollama", "beam", "Beam:Cloud")
    assert not same_ollama_model("ollama-cloud", "glm-5.3", "glm-5.3-flash")
    assert not same_ollama_model("anthropic", "glm-5.3-flash:cloud", "glm-5.3-flash")


def test_ollama_cloud_pick_leaves_later_providers_untouched():
    """The current row's fold must not disturb the rows after it."""
    rows = [
        _row("ollama-cloud", ["glm-5.3-flash"]),
        _row("nous", ["hermes-4-405b"]),
    ]
    rows[1]["is_current"] = False

    out = _finalize_picker_rows(rows, {}, "glm-5.3-flash:cloud")

    by_slug = {r["slug"]: r for r in out}
    assert set(by_slug) == {"ollama-cloud", "nous"}
    assert by_slug["ollama-cloud"]["models"] == ["glm-5.3-flash"]
    assert by_slug["nous"]["models"] == ["hermes-4-405b"]
    assert by_slug["nous"]["total_models"] == 1


def test_real_ollama_hyphen_cloud_tag_is_a_distinct_model():
    """``qwen3-coder:480b-cloud`` is a REAL ollama cloud tag, not a spelling of
    ``qwen3-coder:480b`` — only a trailing ``:cloud`` folds."""
    assert not same_ollama_model("ollama-cloud", "qwen3-coder:480b-cloud", "qwen3-coder:480b")
    assert same_ollama_model("ollama-cloud", "qwen3-coder:480b:cloud", "qwen3-coder:480b")


def test_empty_and_none_ids_never_name_the_same_model():
    """Two absent ids are not one model: an empty pick must match nothing."""
    assert not same_ollama_model("ollama-cloud", "", "")
    assert not same_ollama_model("ollama-cloud", None, "")
    assert not same_ollama_model("ollama-cloud", "", "glm-5.3-flash")
    assert not same_ollama_model("ollama-cloud", "glm-5.3-flash", "")
    assert not same_ollama_model("", "glm-5.3-flash", "glm-5.3-flash")


def test_folding_is_case_insensitive():
    assert same_ollama_model("ollama-cloud", "GLM-5.3-FLASH:Cloud", "glm-5.3-flash")
    assert same_ollama_model("ollama-cloud", "GLM-5.3-FLASH", "glm-5.3-flash:cloud")