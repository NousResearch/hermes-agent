import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure the backend app package resolves in tests
BACKEND_ROOT = Path(__file__).resolve().parents[1]
os.chdir(BACKEND_ROOT)
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

_tmp_dir = tempfile.TemporaryDirectory(prefix="ikb-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp_dir.name) / 'test.db'}"
os.environ["JWT_SECRET"] = "test-secret-0123456789abcdef-0123456789abcdef"
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "720"

from app.main import app
from app.database import Base, SessionLocal, engine


@pytest.fixture(scope="session", autouse=True)
def _setup_db():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def db_session():
    with SessionLocal() as db:
        yield db


@pytest.fixture()
def auth_headers(client):
    email = "admin@example.com"
    password = "Password123!"

    reg = client.post(
        "/api/auth/register",
        json={
            "company_name": "Acme Oy",
            "name": "Admin",
            "email": email,
            "password": password,
        },
    )
    if reg.status_code not in (200, 409):
        raise AssertionError(reg.text)

    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
