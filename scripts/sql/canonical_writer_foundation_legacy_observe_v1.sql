-- Canonical Writer legacy truth observation for the isolated v2 canary.
--
-- This read-only artifact is invoked only after the general observation has
-- proven the exact legacy nineteen-column shape.  The reconciliation artifact
-- independently revalidates the complete relation, ACL, index, and row hash
-- contract before moving any object.

BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY;
SET LOCAL TimeZone = 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '5min';

SELECT pg_catalog.pg_advisory_xact_lock_shared(4841739663211427921);
LOCK TABLE public.canonical_event_log IN ACCESS SHARE MODE;

WITH row_receipts AS (
    SELECT event.event_id,
           pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               pg_catalog.jsonb_build_object(
                   'event_id', pg_catalog.to_jsonb(event)->'event_id',
                   'schema_version', pg_catalog.to_jsonb(event)->'schema_version',
                   'event_type', pg_catalog.to_jsonb(event)->'event_type',
                   'occurred_at', pg_catalog.to_jsonb(event)->'occurred_at',
                   'case_id', pg_catalog.to_jsonb(event)->'case_id',
                   'source', pg_catalog.to_jsonb(event)->'source',
                   'actor', pg_catalog.to_jsonb(event)->'actor',
                   'subject', pg_catalog.to_jsonb(event)->'subject',
                   'evidence', pg_catalog.to_jsonb(event)->'evidence',
                   'decision', pg_catalog.to_jsonb(event)->'decision',
                   'status', pg_catalog.to_jsonb(event)->'status',
                   'next_action', pg_catalog.to_jsonb(event)->'next_action',
                   'safety', pg_catalog.to_jsonb(event)->'safety',
                   'payload', pg_catalog.to_jsonb(event)->'payload'
               )::text, 'UTF8'
           )), 'hex') AS canonical14_row_sha256,
           pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               pg_catalog.to_jsonb(event)::text, 'UTF8'
           )), 'hex') AS extended19_row_sha256
      FROM public.canonical_event_log AS event
), snapshot AS (
    SELECT pg_catalog.count(*)::bigint AS source_row_count,
           pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               'canonical-writer-legacy-reconcile-v1:canonical14' || E'\n'
               || COALESCE(pg_catalog.string_agg(
                    event_id::text || ':' || canonical14_row_sha256,
                    E'\n' ORDER BY event_id
               ), ''), 'UTF8'
           )), 'hex') AS canonical14_sha256,
           pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               'canonical-writer-legacy-reconcile-v1:extended19' || E'\n'
               || COALESCE(pg_catalog.string_agg(
                    event_id::text || ':' || extended19_row_sha256,
                    E'\n' ORDER BY event_id
               ), ''), 'UTF8'
           )), 'hex') AS extended19_sha256
      FROM row_receipts
), cutoffs AS (
    SELECT max(occurred_at) AS occurred_at_cutoff,
           max(inserted_at) AS inserted_at_cutoff
      FROM public.canonical_event_log
), bridge_membership AS (
    SELECT pg_catalog.count(*)::integer AS membership_count,
           COALESCE(pg_catalog.bool_and(membership.admin_option), false)
               AS admin_option,
           COALESCE(pg_catalog.bool_and(membership.inherit_option), false)
               AS inherit_option,
           COALESCE(pg_catalog.bool_and(membership.set_option), false)
               AS set_option
      FROM pg_catalog.pg_auth_members AS membership
      JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
      JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
     WHERE granted.rolname = (
            SELECT pg_catalog.pg_get_userbyid(class.relowner)
              FROM pg_catalog.pg_class AS class
             WHERE class.oid = 'public.canonical_event_log'::regclass
       )
       AND member.rolname = SESSION_USER
)
SELECT pg_catalog.jsonb_build_object(
    'source_owner', (
        SELECT pg_catalog.pg_get_userbyid(class.relowner)
          FROM pg_catalog.pg_class AS class
         WHERE class.oid = 'public.canonical_event_log'::regclass
    ),
    'source_row_count', snapshot.source_row_count,
    'canonical14_sha256', snapshot.canonical14_sha256,
    'extended19_sha256', snapshot.extended19_sha256,
    'occurred_at_cutoff', cutoffs.occurred_at_cutoff,
    'inserted_at_cutoff', cutoffs.inserted_at_cutoff
    , 'bridge_admin', SESSION_USER
    , 'bridge_admin_option', bridge_membership.admin_option
    , 'bridge_inherit_option', bridge_membership.inherit_option
    , 'bridge_set_option', bridge_membership.set_option
    , 'bridge_membership_count', bridge_membership.membership_count
)::text AS legacy_truth_observation
FROM snapshot CROSS JOIN cutoffs CROSS JOIN bridge_membership;

COMMIT;
