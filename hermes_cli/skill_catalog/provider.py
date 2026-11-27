"""Synchronous provider contract: discovery never supplies an install bundle."""


class SkillCatalogProvider:
    """Experimental catalog API. Every method takes one core-stamped request.

    Implement effects with the owning context's ``skill_catalog_mutations``
    facade. Tokens and operation receipts belong to the provider, not core.
    Future optional methods have unsupported defaults for additive evolution.
    """

    name = ""
    label = ""
    implementation_revision = ""

    def availability(self, request: dict) -> dict:
        return {"ok": True, "data": {"state": "ready", "message": ""}}

    def search(self, request: dict) -> dict:
        return self._unsupported()

    def inspect(self, request: dict) -> dict:
        return self._unsupported()

    def scan(self, request: dict) -> dict:
        return self._unsupported()

    def installed(self, request: dict) -> dict:
        return self._unsupported()

    def check(self, request: dict) -> dict:
        return self._unsupported()

    def review(self, request: dict) -> dict:
        return self._unsupported()

    def reviewed(self, request: dict) -> dict:
        return self._unsupported()

    def confirm(self, request: dict) -> dict:
        return self._unsupported()

    def operation(self, request: dict) -> dict:
        return self._unsupported()

    @staticmethod
    def _unsupported() -> dict:
        return {"ok": False, "error": {"code": "unsupported", "message": "Method is unsupported."}}
