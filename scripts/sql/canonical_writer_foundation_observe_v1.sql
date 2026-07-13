-- Canonical Writer persistent-foundation observation v1.
--
-- This is a read-only, PostgreSQL 18, isolated-canary observation.  It has no
-- parameter that can select a host, database, role, schema, or deployment
-- scope.  The caller separately binds the verified TLS peer certificate.

BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY;
SET LOCAL TimeZone = 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '2min';

SELECT pg_catalog.pg_advisory_xact_lock_shared(4841739663211427921);

WITH target_roles AS (
    SELECT role.rolname AS name,
           role.rolcanlogin AS can_login,
           role.rolinherit AS inherits,
           role.rolsuper AS superuser,
           role.rolcreatedb AS create_database,
           role.rolcreaterole AS create_role,
           role.rolreplication AS replication,
           role.rolbypassrls AS bypass_row_security,
           role.rolconnlimit AS connection_limit,
           role.rolvaliduntil IS NULL AS validity_is_unbounded,
           role.rolconfig IS NULL AS configuration_is_empty
      FROM pg_catalog.pg_roles AS role
     WHERE role.rolname IN (
        'canonical_brain_migration_owner',
        'canonical_brain_writer',
        'canonical_brain_canary_bootstrap',
        'canonical_brain_canary_bootstrap_login',
        'muncho_canary_writer_login'
     )
), target_memberships AS (
    SELECT granted.rolname AS granted_role,
           member.rolname AS member_role,
           membership.admin_option,
           membership.inherit_option,
           membership.set_option
      FROM pg_catalog.pg_auth_members AS membership
      JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
      JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
     WHERE granted.rolname IN (
                'canonical_brain_migration_owner',
                'canonical_brain_writer',
                'canonical_brain_canary_bootstrap',
                'canonical_brain_canary_bootstrap_login',
                'muncho_canary_writer_login'
           )
        OR member.rolname IN (
                'canonical_brain_migration_owner',
                'canonical_brain_writer',
                'canonical_brain_canary_bootstrap',
                'canonical_brain_canary_bootstrap_login',
                'muncho_canary_writer_login'
           )
), event_columns AS (
    SELECT pg_catalog.array_agg(
               attribute.attname || ':'
               || pg_catalog.format_type(attribute.atttypid, attribute.atttypmod)
               || ':' || attribute.attnotnull::text
               ORDER BY attribute.attnum
           ) AS identity
      FROM pg_catalog.pg_attribute AS attribute
     WHERE attribute.attrelid = pg_catalog.to_regclass(
               'public.canonical_event_log'
           )
       AND attribute.attnum > 0
       AND NOT attribute.attisdropped
), event_identity AS (
    SELECT CASE
             WHEN pg_catalog.to_regclass('public.canonical_event_log') IS NULL
               THEN 'absent'
             WHEN columns.identity = ARRAY[
                'event_id:uuid:true',
                'schema_version:text:true',
                'event_type:text:true',
                'occurred_at:timestamp with time zone:true',
                'case_id:text:true',
                'source:jsonb:true',
                'actor:jsonb:true',
                'subject:jsonb:true',
                'evidence:jsonb:true',
                'decision:jsonb:true',
                'status:jsonb:true',
                'next_action:jsonb:true',
                'safety:jsonb:true',
                'payload:jsonb:true'
             ]::text[] THEN 'canonical14'
             WHEN columns.identity = ARRAY[
                'event_id:uuid:true',
                'schema_version:text:true',
                'event_type:text:true',
                'occurred_at:timestamp with time zone:true',
                'case_id:text:true',
                'source:jsonb:true',
                'actor:jsonb:true',
                'subject:jsonb:true',
                'evidence:jsonb:true',
                'decision:jsonb:true',
                'status:jsonb:true',
                'next_action:jsonb:true',
                'safety:jsonb:true',
                'payload:jsonb:true',
                'inserted_at:timestamp with time zone:true',
                'idempotency_key:text:false',
                'source_spool:text:false',
                'spool_line_number:integer:false',
                'raw_event_sha256:text:false'
             ]::text[] THEN 'legacy19'
             ELSE 'invalid'
           END AS shape,
           CASE WHEN pg_catalog.to_regclass('public.canonical_event_log') IS NULL
                THEN NULL
                ELSE (
                    SELECT pg_catalog.pg_get_userbyid(class.relowner)
                      FROM pg_catalog.pg_class AS class
                     WHERE class.oid = 'public.canonical_event_log'::regclass
                )
           END AS owner_name
      FROM event_columns AS columns
), database_acl AS (
    SELECT COALESCE(
               pg_catalog.jsonb_agg(
                   pg_catalog.jsonb_build_object(
                       'grantee', CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                            ELSE pg_catalog.pg_get_userbyid(acl.grantee) END,
                       'grantor', pg_catalog.pg_get_userbyid(acl.grantor),
                       'privilege', acl.privilege_type,
                       'grantable', acl.is_grantable
                   ) ORDER BY
                       CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                            ELSE pg_catalog.pg_get_userbyid(acl.grantee) END,
                       acl.privilege_type
               ), '[]'::jsonb
           ) AS value
      FROM pg_catalog.pg_database AS database
      CROSS JOIN LATERAL pg_catalog.aclexplode(
          COALESCE(database.datacl, pg_catalog.acldefault('d', database.datdba))
      ) AS acl
     WHERE database.datname = pg_catalog.current_database()
       AND acl.grantee <> database.datdba
), public_acl AS (
    SELECT COALESCE(
               pg_catalog.jsonb_agg(
                   pg_catalog.jsonb_build_object(
                       'grantee', CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                            ELSE pg_catalog.pg_get_userbyid(acl.grantee) END,
                       'grantor', pg_catalog.pg_get_userbyid(acl.grantor),
                       'privilege', acl.privilege_type,
                       'grantable', acl.is_grantable
                   ) ORDER BY
                       CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                            ELSE pg_catalog.pg_get_userbyid(acl.grantee) END,
                       acl.privilege_type
               ), '[]'::jsonb
           ) AS value
      FROM pg_catalog.pg_namespace AS namespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(
          COALESCE(namespace.nspacl, pg_catalog.acldefault('n', namespace.nspowner))
      ) AS acl
     WHERE namespace.nspname = 'public'
       AND acl.grantee <> namespace.nspowner
)
SELECT pg_catalog.jsonb_build_object(
    'database', pg_catalog.current_database(),
    'postgres_version_num',
        pg_catalog.current_setting('server_version_num')::integer,
    'session_user', SESSION_USER,
    'current_user', CURRENT_USER,
    'roles', COALESCE((
        SELECT pg_catalog.jsonb_agg(
                   pg_catalog.to_jsonb(role_row) ORDER BY role_row.name
               )
          FROM target_roles AS role_row
    ), '[]'::jsonb),
    'memberships', COALESCE((
        SELECT pg_catalog.jsonb_agg(
                   pg_catalog.to_jsonb(membership_row)
                   ORDER BY membership_row.granted_role,
                            membership_row.member_role
               )
          FROM target_memberships AS membership_row
    ), '[]'::jsonb),
    'event_log_shape', event.shape,
    'event_log_owner', event.owner_name,
    'legacy_archive_present', pg_catalog.to_regclass(
        'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'
    ) IS NOT NULL,
    'canonical_schema_owner', (
        SELECT pg_catalog.pg_get_userbyid(namespace.nspowner)
          FROM pg_catalog.pg_namespace AS namespace
         WHERE namespace.nspname = 'canonical_brain'
    ),
    'writer_ping_present', pg_catalog.to_regprocedure(
        'canonical_brain.writer_ping(jsonb,jsonb)'
    ) IS NOT NULL,
    'database_acl', database_acl.value,
    'public_schema_acl', public_acl.value
)::text AS foundation_observation
FROM event_identity AS event
CROSS JOIN database_acl
CROSS JOIN public_acl;

COMMIT;
