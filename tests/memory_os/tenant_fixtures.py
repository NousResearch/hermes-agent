"""Demo fixtures for multi-tenant memory isolation tests.

These fixtures use synthetic data only - no real user information.
"""
from dataclasses import dataclass, field
from typing import Optional

# === Demo Tokens (unique identifiers for isolation verification) ===
ALICE_ONLY_TOKEN = "ALICE_ONLY_7391"
BOB_ONLY_TOKEN = "BOB_ONLY_4826"
COMMON_RULE_TOKEN = "COMMON_RULE_0001"

# === Demo Namespaces ===
ALICE_NAMESPACE = "test_user_alice"
BOB_NAMESPACE = "test_user_bob"
CORE_NAMESPACE = ""

# === Demo Data ===
@dataclass
class DemoUser:
    namespace: str
    display_name: str
    secret_token: str
    is_admin: bool = False

ALICE = DemoUser(
    namespace=ALICE_NAMESPACE,
    display_name="Alice",
    secret_token=ALICE_ONLY_TOKEN,
    is_admin=False,
)

BOB = DemoUser(
    namespace=BOB_NAMESPACE,
    display_name="Bob",
    secret_token=BOB_ONLY_TOKEN,
    is_admin=False,
)

ADMIN = DemoUser(
    namespace=CORE_NAMESPACE,
    display_name="Admin",
    secret_token="ADMIN_TOKEN",
    is_admin=True,
)

# === Demo Facts ===
ALICE_FACTS = [
    {
        "path": "用户档案/Alice私有",
        "content": f"{ALICE_ONLY_TOKEN} Alice likes blue. Private profile.",
        "namespace": ALICE_NAMESPACE,
        "priority": 8,
    },
    {
        "path": "项目/ProjectX",
        "content": f"{ALICE_ONLY_TOKEN} ProjectX uses React and PostgreSQL.",
        "namespace": ALICE_NAMESPACE,
        "priority": 7,
    },
]

BOB_FACTS = [
    {
        "path": "用户档案/Bob私有",
        "content": f"{BOB_ONLY_TOKEN} Bob likes red. Private profile.",
        "namespace": BOB_NAMESPACE,
        "priority": 8,
    },
    {
        "path": "项目/ProjectY",
        "content": f"{BOB_ONLY_TOKEN} ProjectY uses Vue and SQLite.",
        "namespace": BOB_NAMESPACE,
        "priority": 7,
    },
]

CORE_RULES = [
    {
        "path": "规则/CommonRule",
        "content": f"{COMMON_RULE_TOKEN} Always be helpful and accurate.",
        "namespace": CORE_NAMESPACE,
        "priority": 10,
    },
]

# === Setup/Teardown Helpers ===
async def setup_fixtures(graph_service, db_session):
    """Create demo tenant data in the database."""
    from sqlalchemy import text
    
    # Create Alice's data
    for fact in ALICE_FACTS:
        try:
            await graph_service.create_memory(
                parent_path=fact["path"].rsplit("/", 1)[0] if "/" in fact["path"] else "",
                content=fact["content"],
                domain="core",
                namespace=fact["namespace"],
                title=fact["path"].split("/")[-1],
                priority=fact["priority"],
                disclosure="demo_fixture",
            )
        except Exception as e:
            pass  # May already exist
    
    # Create Bob's data
    for fact in BOB_FACTS:
        try:
            await graph_service.create_memory(
                parent_path=fact["path"].rsplit("/", 1)[0] if "/" in fact["path"] else "",
                content=fact["content"],
                domain="core",
                namespace=fact["namespace"],
                title=fact["path"].split("/")[-1],
                priority=fact["priority"],
                disclosure="demo_fixture",
            )
        except Exception as e:
            pass

async def teardown_fixtures(db_session):
    """Remove all demo tenant data."""
    from sqlalchemy import text
    for ns in [ALICE_NAMESPACE, BOB_NAMESPACE]:
        await db_session.execute(
            text("DELETE FROM mg_paths WHERE namespace = :ns"), {"ns": ns}
        )
    # Clean up nodes created for fixtures
    await db_session.execute(text("""
        DELETE FROM mg_memories WHERE node_uuid IN (
            SELECT uuid FROM mg_nodes WHERE uuid NOT IN (
                SELECT DISTINCT node_uuid FROM mg_paths WHERE namespace NOT IN (:ns1, :ns2)
            )
        )
    """), {"ns1": ALICE_NAMESPACE, "ns2": BOB_NAMESPACE})
