"""Container sandbox image defaults, split out of ``hermes_cli.config_defaults``.

Pure-data leaf module (same rule: no imports from ``hermes_cli.config``); the names are
re-exported by ``config_defaults`` so existing import sites are unchanged.
"""


#: Image every container terminal backend (docker/modal/daytona/singularity) uses unless the
#: user pins one. LEGACY_SANDBOX_IMAGES are the plain defaults that preceded the desktop stack
#: (the 3.14 pin shipped between the two without a migration); a saved config still holding one
#: is the template copied, and the config migration unsets it, never a user's own pin.
DEFAULT_SANDBOX_IMAGE = "nousresearch/hermes-sandbox:desktop"
LEGACY_SANDBOX_IMAGES = ("nikolaik/python-nodejs:python3.11-nodejs20", "nikolaik/python-nodejs:python3.14-nodejs22")
LEGACY_SANDBOX_IMAGE = LEGACY_SANDBOX_IMAGES[0]
# Vercel Sandbox managed image (Vercel deprecated its `runtime` presets in Aug 2026).
DEFAULT_VERCEL_IMAGE = "vercel/sandbox/universal:latest"
LEGACY_VERCEL_RUNTIME = "node24"  # the seeded pre-49 default, never a user choice


