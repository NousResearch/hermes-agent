-- Research Protocol PR2: one-shot least-privilege runtime roles.
-- Apply once as a trusted cluster administrator after 0001. A pre-existing role
-- name makes the transaction fail closed; this migration never attempts to
-- sanitize a role that may hold privileges in another database.

BEGIN;

CREATE ROLE research_protocol_planner_reader
    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

CREATE ROLE research_protocol_approval_writer
    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

CREATE ROLE research_protocol_approval_claimant
    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

REVOKE ALL ON SCHEMA research_protocol FROM PUBLIC;
REVOKE ALL ON research_protocol.approvals FROM PUBLIC;
REVOKE ALL ON FUNCTION research_protocol.store_approval(
    text, text, text, text, text, text, timestamptz, timestamptz, integer
) FROM PUBLIC;
REVOKE ALL ON FUNCTION research_protocol.claim_approval(text, text, text)
    FROM PUBLIC;

REVOKE ALL ON SCHEMA research_protocol FROM research_protocol_planner_reader;
REVOKE ALL ON SCHEMA research_protocol FROM research_protocol_approval_writer;
REVOKE ALL ON SCHEMA research_protocol FROM research_protocol_approval_claimant;

REVOKE ALL ON research_protocol.approvals FROM research_protocol_planner_reader;
REVOKE ALL ON research_protocol.approvals FROM research_protocol_approval_writer;
REVOKE ALL ON research_protocol.approvals FROM research_protocol_approval_claimant;

REVOKE ALL ON FUNCTION research_protocol.store_approval(
    text, text, text, text, text, text, timestamptz, timestamptz, integer
) FROM research_protocol_planner_reader;
REVOKE ALL ON FUNCTION research_protocol.store_approval(
    text, text, text, text, text, text, timestamptz, timestamptz, integer
) FROM research_protocol_approval_writer;
REVOKE ALL ON FUNCTION research_protocol.store_approval(
    text, text, text, text, text, text, timestamptz, timestamptz, integer
) FROM research_protocol_approval_claimant;

REVOKE ALL ON FUNCTION research_protocol.claim_approval(text, text, text)
    FROM research_protocol_planner_reader;
REVOKE ALL ON FUNCTION research_protocol.claim_approval(text, text, text)
    FROM research_protocol_approval_writer;
REVOKE ALL ON FUNCTION research_protocol.claim_approval(text, text, text)
    FROM research_protocol_approval_claimant;

GRANT USAGE ON SCHEMA research_protocol
    TO research_protocol_planner_reader;
GRANT SELECT ON research_protocol.approvals TO research_protocol_planner_reader;

GRANT USAGE ON SCHEMA research_protocol
    TO research_protocol_approval_writer;
GRANT EXECUTE ON FUNCTION research_protocol.store_approval(
    text, text, text, text, text, text, timestamptz, timestamptz, integer
) TO research_protocol_approval_writer;

GRANT USAGE ON SCHEMA research_protocol
    TO research_protocol_approval_claimant;
GRANT EXECUTE ON FUNCTION research_protocol.claim_approval(text, text, text)
    TO research_protocol_approval_claimant;

COMMIT;
