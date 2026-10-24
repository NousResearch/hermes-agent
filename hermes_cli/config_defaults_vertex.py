"""The ``vertex:`` section of ``config_defaults.DEFAULT_CONFIG`` (pure-data leaf, no imports).

Split out of ``hermes_cli/config_defaults.py`` to keep that facade's file size at its cap;
the section is the Vertex provider's own knobs and grows with the provider.
"""

# Google Vertex AI (Gemini). Auth is OAuth2 from a service-account JSON or ADC, NOT an API key;
# the credential path lives in .env (VERTEX_CREDENTIALS_PATH / GOOGLE_APPLICATION_CREDENTIALS).
# Bridged to VERTEX_PROJECT_ID / VERTEX_REGION.
VERTEX_CONFIG = {
    # GCP project ID. Empty -> project_id from the service-account JSON (or ADC).
    "project_id": "",
    # "global" is required for Gemini 3.x preview models (regional endpoints silently 404); use
    # e.g. "us-central1" only if your models are region-pinned.
    "region": "global",
    # Opt in to Priority PayGo: the request carries
    # X-Vertex-AI-LLM-Shared-Request-Type: priority, which trades a higher per-token rate for
    # steadier service than the shared Standard PayGo pool (which answers capacity 429s).
    # Off by default - it changes what the user pays per token.
    "priority_paygo": False,
}
