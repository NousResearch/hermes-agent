from types import SimpleNamespace

from agent import auxiliary_model_resolution as auxiliary


def test_profile_default_is_the_provider_auxiliary_source(monkeypatch):
    profile = SimpleNamespace(name="provider-x", default_aux_model="cheap-model")
    monkeypatch.setattr("providers.get_provider_profile", lambda _provider: profile)
    assert auxiliary.select_provider_auxiliary_model("provider-x") == "cheap-model"


def test_dynamic_aux_recommendation_is_only_used_for_fast_opt_in(monkeypatch):
    class Profile:
        name = "provider-x"
        default_aux_model = "default-model"

        def resolve_aux_model(self):
            return "dynamic-model"

    monkeypatch.setattr("providers.get_provider_profile", lambda _provider: Profile())
    monkeypatch.setattr(auxiliary, "_fast_catalog_ids", lambda _provider: ())
    assert auxiliary.select_provider_auxiliary_model("provider-x") == "default-model"
    assert (
        auxiliary.select_provider_auxiliary_model("provider-x", prefer_fast=True)
        == "dynamic-model"
    )


def test_nous_policy_rejects_blocked_aux_default(monkeypatch):
    profile = SimpleNamespace(name="nous", default_aux_model="blocked/model")
    monkeypatch.setattr("providers.get_provider_profile", lambda _provider: profile)
    monkeypatch.setattr(auxiliary, "_nous_allowed_ids", lambda: {"allowed/model"})
    assert auxiliary.select_provider_auxiliary_model("nous") == ""


def test_declarative_provider_aux_and_vision_defaults_are_registered():
    from providers import get_provider_profile

    assert get_provider_profile("tencent-tokenhub").default_aux_model == "hy4-preview"
    assert get_provider_profile("tencent-tokenplan").default_aux_model == "hy4-preview"
    assert get_provider_profile("xiaomi").default_vision_model() == "mimo-v2.5"
    assert get_provider_profile("zai").default_vision_model() == "glm-5.3-flash"


def test_provider_profiles_own_vision_rejection():
    from providers import get_provider_profile

    assert get_provider_profile("kimi-coding").rejects_vision_input is True
    assert get_provider_profile("kimi-coding-cn").rejects_vision_input is True


def test_auxiliary_resolution_exports_no_vision_semantic_adapters():
    for obsolete in (
        "provider_rejects_vision_input",
        "provider_vision_default",
        "is_declared_vision_default",
        "select_provider_vision_model",
    ):
        assert not hasattr(auxiliary, obsolete)
