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
), database_identity AS (
    SELECT pg_catalog.pg_get_userbyid(database.datdba) AS owner_name
      FROM pg_catalog.pg_database AS database
     WHERE database.datname = pg_catalog.current_database()
), target_memberships AS (
    SELECT granted.rolname AS granted_role,
           member.rolname AS member_role,
           grantor.rolname AS grantor,
           membership.admin_option,
           membership.inherit_option,
           membership.set_option
      FROM pg_catalog.pg_auth_members AS membership
      JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
      JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
      JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = membership.grantor
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
), event_relation AS (
    SELECT class.oid,
           pg_catalog.pg_get_userbyid(class.relowner) AS owner_name
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
     WHERE namespace.nspname = 'public'
       AND class.relname = 'canonical_event_log'
), event_columns AS (
    SELECT pg_catalog.array_agg(
               attribute.attname || ':'
               || pg_catalog.format_type(attribute.atttypid, attribute.atttypmod)
               || ':' || attribute.attnotnull::text
               ORDER BY attribute.attnum
           ) AS identity
      FROM pg_catalog.pg_attribute AS attribute
      JOIN event_relation AS relation ON relation.oid = attribute.attrelid
     WHERE attribute.attnum > 0
       AND NOT attribute.attisdropped
), event_identity AS (
    SELECT CASE
             WHEN relation.oid IS NULL
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
           relation.owner_name
      FROM event_columns AS columns
      LEFT JOIN event_relation AS relation ON true
), legacy_archive_relation AS (
    SELECT class.oid,
           owner_role.rolname AS owner_name,
           class.relkind,
           class.relpersistence,
           owner_role.rolsuper AS owner_superuser,
           owner_role.rolcreatedb AS owner_create_database,
           owner_role.rolcreaterole AS owner_create_role,
           owner_role.rolreplication AS owner_replication,
           owner_role.rolbypassrls AS owner_bypass_row_security,
           owner_role.rolconnlimit AS owner_connection_limit,
           owner_role.rolvaliduntil IS NULL AS owner_validity_is_unbounded,
           owner_role.rolconfig IS NULL AS owner_configuration_is_empty
      FROM pg_catalog.pg_class AS class
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = class.relnamespace
      JOIN pg_catalog.pg_roles AS owner_role
        ON owner_role.oid = class.relowner
     WHERE namespace.nspname = 'canonical_brain_legacy_quarantine'
       AND class.relname = 'canonical_event_log_legacy_v1'
), legacy_archive AS (
    SELECT pg_catalog.count(*) = 1 AS present,
           (
               SELECT pg_catalog.jsonb_build_object(
                          'oid', relation.oid::text,
                          'owner', relation.owner_name,
                          'relation_kind', relation.relkind::text,
                          'persistence', relation.relpersistence::text,
                          'owner_superuser', relation.owner_superuser,
                          'owner_create_database',
                              relation.owner_create_database,
                          'owner_create_role', relation.owner_create_role,
                          'owner_replication', relation.owner_replication,
                          'owner_bypass_row_security',
                              relation.owner_bypass_row_security,
                          'owner_connection_limit',
                              relation.owner_connection_limit,
                          'owner_validity_is_unbounded',
                              relation.owner_validity_is_unbounded,
                          'owner_configuration_is_empty',
                              relation.owner_configuration_is_empty
                      )
                 FROM legacy_archive_relation AS relation
           ) AS identity
      FROM legacy_archive_relation
), writer_ping_identity AS (
    SELECT pg_catalog.count(*) = 1 AS present
      FROM pg_catalog.pg_proc AS procedure
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = procedure.pronamespace
      JOIN pg_catalog.pg_type AS jsonb_type
        ON jsonb_type.typname = 'jsonb'
      JOIN pg_catalog.pg_namespace AS type_namespace
        ON type_namespace.oid = jsonb_type.typnamespace
       AND type_namespace.nspname = 'pg_catalog'
     WHERE namespace.nspname = 'canonical_brain'
       AND procedure.proname = 'writer_ping'
       AND procedure.pronargs = 2
       AND procedure.proargtypes[0] = jsonb_type.oid
       AND procedure.proargtypes[1] = jsonb_type.oid
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
    'database_owner', database_identity.owner_name,
    'postgres_version_num',
        pg_catalog.current_setting('server_version_num')::integer,
    'session_user', SESSION_USER,
    'current_user', CURRENT_USER,
    'temporary_admin_roles', COALESCE((
        SELECT pg_catalog.jsonb_agg(role.rolname ORDER BY role.rolname)
          FROM pg_catalog.pg_roles AS role
         WHERE role.rolname ~ '^muncho_canary_admin_[0-9a-f]{16}$'
    ), '[]'::jsonb),
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
    'legacy_archive_present', legacy_archive.present,
    'legacy_archive_identity', legacy_archive.identity,
    'canonical_schema_owner', (
        SELECT pg_catalog.pg_get_userbyid(namespace.nspowner)
          FROM pg_catalog.pg_namespace AS namespace
         WHERE namespace.nspname = 'canonical_brain'
    ),
    'writer_ping_present', writer_ping_identity.present,
    'database_acl', database_acl.value,
    'public_schema_acl', public_acl.value
)::text AS foundation_observation
FROM event_identity AS event
CROSS JOIN database_identity
CROSS JOIN legacy_archive
CROSS JOIN writer_ping_identity
CROSS JOIN database_acl
CROSS JOIN public_acl;

COMMIT;
