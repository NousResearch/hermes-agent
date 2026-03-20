from hermes_cli.product_users import (
    create_product_signup_token,
    create_product_user,
    deactivate_product_user,
    get_product_user_by_id,
    list_product_users,
)


class DummyResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)
        self.content = b"{}"

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        status_code, payload = self.responses[(method, path)]
        return DummyResponse(status_code, payload)

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)


def test_list_product_users_filters_internal_and_placeholder_email(monkeypatch):
    client = DummyClient(
        {
            (
                "GET",
                "/api/users",
            ): (
                200,
                {
                    "data": [
                        {
                            "id": "internal",
                            "username": "static-api-user-abc",
                            "email": None,
                            "emailVerified": False,
                            "firstName": "Static",
                            "lastName": "API",
                            "displayName": "Static API",
                            "isAdmin": True,
                            "locale": None,
                            "customClaims": [],
                            "userGroups": [],
                            "ldapId": None,
                            "disabled": False,
                        },
                        {
                            "id": "user-1",
                            "username": "maria",
                            "email": "maria@users.local.invalid",
                            "emailVerified": False,
                            "firstName": "Maria",
                            "lastName": "User",
                            "displayName": "Maria User",
                            "isAdmin": False,
                            "locale": None,
                            "customClaims": [],
                            "userGroups": [],
                            "ldapId": None,
                            "disabled": False,
                        },
                    ]
                },
            )
        }
    )
    monkeypatch.setattr("hermes_cli.product_users._client", lambda config=None: client)

    users = list_product_users()

    assert len(users) == 1
    assert users[0].username == "maria"
    assert users[0].email is None
    assert users[0].email_is_placeholder is True


def test_create_product_user_uses_placeholder_email_when_missing(monkeypatch):
    client = DummyClient(
        {
            ("POST", "/api/users"): (
                200,
                {
                    "id": "user-1",
                    "username": "maria",
                    "email": "maria@users.local.invalid",
                    "emailVerified": False,
                    "firstName": "Maria",
                    "lastName": "Example",
                    "displayName": "Maria Example",
                    "isAdmin": False,
                    "locale": None,
                    "customClaims": [],
                    "userGroups": [],
                    "ldapId": None,
                    "disabled": False,
                },
            )
        }
    )
    monkeypatch.setattr("hermes_cli.product_users._client", lambda config=None: client)

    user = create_product_user("maria", "Maria Example")

    assert user.username == "maria"
    assert user.email is None
    payload = client.calls[0][2]["json"]
    assert payload["email"] == "maria@users.local.invalid"


def test_get_product_user_by_id_returns_none_for_missing(monkeypatch):
    client = DummyClient({("GET", "/api/users/user-1"): (404, {"error": "missing"})})
    monkeypatch.setattr("hermes_cli.product_users._client", lambda config=None: client)

    assert get_product_user_by_id("user-1") is None


def test_deactivate_product_user_sets_disabled(monkeypatch):
    client = DummyClient(
        {
            ("GET", "/api/users/user-1"): (
                200,
                {
                    "id": "user-1",
                    "username": "maria",
                    "email": "maria@example.com",
                    "emailVerified": False,
                    "firstName": "Maria",
                    "lastName": "Example",
                    "displayName": "Maria Example",
                    "isAdmin": False,
                    "locale": None,
                    "customClaims": [],
                    "userGroups": [],
                    "ldapId": None,
                    "disabled": False,
                },
            ),
            ("PUT", "/api/users/user-1"): (
                200,
                {
                    "id": "user-1",
                    "username": "maria",
                    "email": "maria@example.com",
                    "emailVerified": False,
                    "firstName": "Maria",
                    "lastName": "Example",
                    "displayName": "Maria Example",
                    "isAdmin": False,
                    "locale": None,
                    "customClaims": [],
                    "userGroups": [],
                    "ldapId": None,
                    "disabled": True,
                },
            ),
        }
    )
    monkeypatch.setattr("hermes_cli.product_users._client", lambda config=None: client)

    user = deactivate_product_user("user-1")

    assert user.disabled is True
    assert client.calls[1][2]["json"]["disabled"] is True


def test_create_product_signup_token_returns_full_url(monkeypatch):
    client = DummyClient({("POST", "/api/signup-tokens"): (200, {"token": "signup-123"})})
    monkeypatch.setattr("hermes_cli.product_users._client", lambda config=None: client)
    monkeypatch.setattr(
        "hermes_cli.product_users.resolve_product_urls",
        lambda config=None: {"issuer_url": "http://localhost:1411"},
    )

    token = create_product_signup_token({})

    assert token.token == "signup-123"
    assert token.signup_url == "http://localhost:1411/st/signup-123"
