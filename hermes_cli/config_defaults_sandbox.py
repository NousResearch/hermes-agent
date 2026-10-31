"""Sandbox image defaults referenced by DEFAULT_CONFIG and migrations.

DEFAULT_SANDBOX_IMAGE is what every container terminal backend (docker/modal/daytona,
the singularity plugin) uses unless the user pins one. LEGACY_SANDBOX_IMAGES are the
plain defaults that preceded the desktop stack (the 3.14 pin shipped between the two
without a migration); a saved config still holding one is the template copied, and
the config migration unsets it, never a user's own pin.

DEFAULT_VERCEL_IMAGE is the Vercel Sandbox managed image (Vercel deprecated its
``runtime`` presets in Aug 2026). LEGACY_VERCEL_RUNTIME is the seeded pre-49 default,
never a user choice.
"""

DEFAULT_SANDBOX_IMAGE = "nousresearch/hermes-sandbox:desktop"
LEGACY_SANDBOX_IMAGES = (
    "nikolaik/python-nodejs:python3.11-nodejs20",
    "nikolaik/python-nodejs:python3.14-nodejs22",
)
LEGACY_SANDBOX_IMAGE = LEGACY_SANDBOX_IMAGES[0]
DEFAULT_VERCEL_IMAGE = "vercel/sandbox/universal:latest"
LEGACY_VERCEL_RUNTIME = "node24"
