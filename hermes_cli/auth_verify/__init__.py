"""One live check of an API key before ``hermes auth add`` stores it.

Modules:
- ``request``: builds the single authenticated read that proves a key works at an endpoint
  (route rewrites, auth header family, configured extra headers), from the same seams chat uses.
- ``check``: runs that read under a whole-call deadline and classifies the answer into a
  ``KeyCheck`` (ok / rejected / billing / unverified), reusing the shared error classifier.
"""

from hermes_cli.auth_verify.check import VERIFY_TIMEOUT_S, KeyCheck, KeyCheckOutcome, verify_api_key

__all__ = ["VERIFY_TIMEOUT_S", "KeyCheck", "KeyCheckOutcome", "verify_api_key"]
