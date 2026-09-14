"""White-label identity and its projection into runtime surfaces."""

from __future__ import annotations

import yaml

from nova.audit import new_correlation_id
from nova.runtime.hermes.skin import build_skin, skin_filename
from nova.spec import IdentitySpec
from nova.spec.identity import DEFAULT_PRODUCT_NAME


def test_unbranded_identity_falls_back_to_the_default_product_name():
    assert IdentitySpec().product_name == DEFAULT_PRODUCT_NAME


def test_product_name_flows_into_the_skin():
    skin = build_skin(IdentitySpec.parse({"product_name": "Acme Intelligence"}), tenant_id="acme")
    assert skin["branding"]["agent_name"] == "Acme Intelligence"
    assert "Acme Intelligence" in skin["branding"]["welcome"]


def test_custom_welcome_wins_over_the_generated_one():
    skin = build_skin(
        IdentitySpec.parse({"product_name": "Acme", "messages": {"welcome": "Hi."}}),
        tenant_id="acme",
    )
    assert skin["branding"]["welcome"] == "Hi."


def test_theme_tokens_map_onto_display_roles():
    skin = build_skin(IdentitySpec.parse({"theme": {"accent": "#1f6f5c"}}), tenant_id="acme")
    assert skin["colors"]["ui_accent"] == "#1f6f5c"


def test_unset_theme_emits_no_colors():
    """A partial identity must not write blanks over the runtime's own defaults."""
    assert "colors" not in build_skin(IdentitySpec(), tenant_id="acme")


def test_agent_display_name_lookup_falls_back():
    identity = IdentitySpec.parse({"agents": {"support": {"display_name": "Acme Helper"}}})
    assert identity.display_name_for("support", "Support") == "Acme Helper"
    assert identity.display_name_for("ops", "Operations") == "Operations"


def test_skin_filename_is_per_tenant():
    assert skin_filename("acme") == "acme.yaml"


def test_apply_identity_writes_a_skin_the_runtime_will_load(home, audit, runtime, bundle):
    runtime.apply_identity(bundle.identity, audit=audit, correlation_id=new_correlation_id())
    written = home / "skins" / "acme.yaml"
    assert written.is_file()
    skin = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert skin["branding"]["agent_name"] == "Acme Intelligence"


def test_apply_identity_is_idempotent(home, audit, runtime, bundle):
    runtime.apply_identity(bundle.identity, audit=audit, correlation_id=new_correlation_id())
    again = runtime.apply_identity(bundle.identity, audit=audit, correlation_id=new_correlation_id())
    assert again.unchanged is True


def test_rebranding_needs_no_code_change(home, audit, runtime, bundle):
    """The whole white-label promise, as one assertion."""
    runtime.apply_identity(bundle.identity, audit=audit, correlation_id=new_correlation_id())
    rebranded = IdentitySpec.parse({"product_name": "Bristol Foods AI Workforce"})
    result = runtime.apply_identity(rebranded, audit=audit, correlation_id=new_correlation_id())
    assert result.changed is True
    skin = yaml.safe_load((home / "skins" / "acme.yaml").read_text(encoding="utf-8"))
    assert skin["branding"]["agent_name"] == "Bristol Foods AI Workforce"


def test_identity_application_is_logged_as_model_visible(home, audit, runtime, bundle):
    runtime.apply_identity(bundle.identity, audit=audit, correlation_id=new_correlation_id())
    events = [e for e in audit.read() if e.kind == "identity.applied"]
    assert [e.phase for e in events] == ["intent", "committed"]
