-- Canonical Writer isolated-canary Phase-B writer preflight v1.
--
-- This is a fixed, catalog-only observation executed as the ordinary writer
-- login.  It neither resolves protected objects through caller privileges nor
-- reads any application relation.  A repeatable-read snapshot binds the
-- existing foundation, either absence or exact recovery-visible bootstrap
-- identities, every connectable-database ACL, and provider-managed authority.
-- Host credential metadata, stopped services, release files, and Cloud SQL
-- users/operations are collected by the outer owner boundary and bound to this
-- nonterminal database receipt before an approval digest can exist.

BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL TimeZone = 'UTC';
SET LOCAL DateStyle = 'ISO, YMD';
SET LOCAL search_path = pg_catalog;
SET LOCAL lock_timeout = '15s';
SET LOCAL statement_timeout = '2min';

DO $phase_b_writer_identity$
BEGIN
    IF pg_catalog.current_database() <> 'muncho_canary_brain'
       OR pg_catalog.current_database() = 'ai_platform_brain'
       OR pg_catalog.current_setting('server_version_num')::integer < 180000
       OR pg_catalog.current_setting('server_version_num')::integer >= 190000
       OR SESSION_USER <> 'muncho_canary_writer_login'
       OR CURRENT_USER <> SESSION_USER THEN
        RAISE EXCEPTION 'phase-b writer preflight identity invalid';
    END IF;
END
$phase_b_writer_identity$;

WITH
target_role_names(name) AS (
    VALUES
      ('canonical_brain_migration_owner'::text),
      ('canonical_brain_writer'::text),
      ('canonical_brain_canary_bootstrap'::text),
      ('canonical_brain_canary_bootstrap_login'::text),
      ('muncho_canary_writer_login'::text)
),
role_identities AS (
    SELECT role.oid,
           role.rolname AS name,
           role.rolcanlogin AS can_login,
           role.rolinherit AS inherits,
           role.rolsuper AS superuser,
           role.rolcreatedb AS create_database,
           role.rolcreaterole AS create_role,
           role.rolreplication AS replication,
           role.rolbypassrls AS bypass_row_security,
           role.rolconnlimit AS connection_limit,
           role.rolvaliduntil IS NULL AS validity_is_unbounded,
           role.rolconfig IS NULL AS configuration_is_empty,
           role.rolname ~ '^muncho_canary_admin_[0-9a-f]{16}$'
               AS temporary_admin
      FROM pg_catalog.pg_roles AS role
     WHERE role.rolname IN (SELECT name FROM target_role_names)
        OR role.rolname ~ '^muncho_canary_admin_[0-9a-f]{16}$'
),
target_roles AS (
    SELECT identity.oid::text AS oid,
           identity.name,
           identity.can_login,
           identity.inherits,
           identity.superuser,
           identity.create_database,
           identity.create_role,
           identity.replication,
           identity.bypass_row_security,
           identity.connection_limit,
           identity.validity_is_unbounded,
           identity.configuration_is_empty
      FROM role_identities AS identity
     WHERE NOT identity.temporary_admin
),
temporary_admin_roles AS (
    SELECT identity.oid::text AS oid,
           identity.name,
           identity.can_login,
           identity.inherits,
           identity.superuser,
           identity.create_database,
           identity.create_role,
           identity.replication,
           identity.bypass_row_security,
           identity.connection_limit,
           identity.validity_is_unbounded,
           identity.configuration_is_empty
      FROM role_identities AS identity
     WHERE identity.temporary_admin
),
observed_memberships AS (
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
     WHERE membership.roleid IN (SELECT oid FROM role_identities)
        OR membership.member IN (SELECT oid FROM role_identities)
        OR membership.grantor IN (SELECT oid FROM role_identities)
),
namespace_scope AS (
    SELECT namespace.oid,
           namespace.nspname AS name,
           owner.oid AS owner_oid,
           owner.rolname AS owner,
           namespace.nspacl,
           namespace.nspowner
      FROM pg_catalog.pg_namespace AS namespace
      JOIN pg_catalog.pg_roles AS owner ON owner.oid = namespace.nspowner
     WHERE namespace.nspname IN (
        'public',
        'canonical_brain',
        'canonical_brain_legacy_quarantine'
     )
),
namespace_acl_rows AS (
    SELECT namespace.oid AS namespace_oid,
           acl.grantor,
           grantor.rolname AS grantor_name,
           acl.grantee,
           CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                ELSE grantee.rolname END AS grantee_name,
           acl.privilege_type,
           acl.is_grantable
      FROM namespace_scope AS namespace
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          namespace.nspacl,
          pg_catalog.acldefault('n', namespace.nspowner)
      )) AS acl
      JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = acl.grantor
      LEFT JOIN pg_catalog.pg_roles AS grantee ON grantee.oid = acl.grantee
),
namespace_observations AS (
    SELECT namespace.name,
           pg_catalog.jsonb_build_object(
               'oid', namespace.oid::text,
               'name', namespace.name,
               'owner_oid', namespace.owner_oid::text,
               'owner', namespace.owner,
               'acl_is_null', namespace.nspacl IS NULL,
               'acl', COALESCE((
                   SELECT pg_catalog.jsonb_agg(
                       pg_catalog.jsonb_build_object(
                           'grantor_oid', acl.grantor::text,
                           'grantor', acl.grantor_name,
                           'grantee_oid', acl.grantee::text,
                           'grantee', acl.grantee_name,
                           'privilege', acl.privilege_type,
                           'grantable', acl.is_grantable
                       ) ORDER BY acl.grantee_name, acl.privilege_type,
                                  acl.grantor_name
                   )
                     FROM namespace_acl_rows AS acl
                    WHERE acl.namespace_oid = namespace.oid
               ), '[]'::jsonb)
           ) AS value
      FROM namespace_scope AS namespace
),
observed_relations AS (
    SELECT target.label,
           namespace.oid AS namespace_oid,
           relation.oid,
           owner.oid AS owner_oid,
           owner.rolname AS owner,
           relation.relkind,
           relation.relpersistence,
           relation.relispartition,
           access_method.amname AS access_method,
           relation.reltablespace,
           relation.relrowsecurity,
           relation.relforcerowsecurity,
           relation.relreplident,
           relation.reloptions,
           relation.relnatts,
           relation.relacl,
           owner.rolsuper AS owner_superuser,
           owner.rolcreatedb AS owner_create_database,
           owner.rolcreaterole AS owner_create_role,
           owner.rolreplication AS owner_replication,
           owner.rolbypassrls AS owner_bypass_row_security,
           owner.rolconnlimit AS owner_connection_limit,
           owner.rolvaliduntil IS NULL AS owner_validity_is_unbounded,
           owner.rolconfig IS NULL AS owner_configuration_is_empty
      FROM (
        VALUES
          ('event_log'::text, 'public'::text, 'canonical_event_log'::text),
          ('legacy_archive'::text, 'canonical_brain_legacy_quarantine'::text,
           'canonical_event_log_legacy_v1'::text)
      ) AS target(label, namespace_name, relation_name)
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.nspname = target.namespace_name
      JOIN pg_catalog.pg_class AS relation
        ON relation.relnamespace = namespace.oid
       AND relation.relname = target.relation_name
      JOIN pg_catalog.pg_roles AS owner ON owner.oid = relation.relowner
      LEFT JOIN pg_catalog.pg_am AS access_method
        ON access_method.oid = relation.relam
),
relation_acl_rows AS (
    SELECT relation.oid AS relation_oid,
           acl.grantor,
           grantor.rolname AS grantor_name,
           acl.grantee,
           CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                ELSE grantee.rolname END AS grantee_name,
           acl.privilege_type,
           acl.is_grantable
      FROM observed_relations AS relation
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          relation.relacl,
          pg_catalog.acldefault('r', relation.owner_oid)
      )) AS acl
      JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = acl.grantor
      LEFT JOIN pg_catalog.pg_roles AS grantee ON grantee.oid = acl.grantee
),
relation_column_acl_rows AS (
    SELECT attribute.attrelid AS relation_oid,
           attribute.attnum,
           acl.grantor,
           grantor.rolname AS grantor_name,
           acl.grantee,
           CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                ELSE grantee.rolname END AS grantee_name,
           acl.privilege_type,
           acl.is_grantable
      FROM observed_relations AS relation
      JOIN pg_catalog.pg_attribute AS attribute
        ON attribute.attrelid = relation.oid
       AND attribute.attnum > 0
       AND NOT attribute.attisdropped
      CROSS JOIN LATERAL pg_catalog.aclexplode(attribute.attacl) AS acl
      JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = acl.grantor
      LEFT JOIN pg_catalog.pg_roles AS grantee ON grantee.oid = acl.grantee
),
relation_columns AS (
    SELECT relation.oid AS relation_oid,
           attribute.attnum::integer AS position,
           attribute.attname AS name,
           attribute.atttypid::text AS type_oid,
           pg_catalog.format_type(
               attribute.atttypid, attribute.atttypmod
           ) AS type,
           attribute.attnotnull AS not_null,
           attribute.atthasdef AS has_default,
           CASE WHEN default_row.oid IS NULL THEN NULL ELSE
               pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
                   pg_catalog.pg_get_expr(
                       default_row.adbin, default_row.adrelid, true
                   ),
                   'UTF8'
               )), 'hex')
           END AS default_expression_sha256,
           attribute.attidentity::text AS identity,
           attribute.attgenerated::text AS generated,
           attribute.atthasmissing AS has_missing,
           attribute.attislocal AS is_local,
           attribute.attinhcount::integer AS inheritance_count,
           attribute.attndims::integer AS array_dimensions,
           attribute.attcollation = data_type.typcollation
               AS collation_is_type_default,
           attribute.attstorage = data_type.typstorage
               AS storage_is_type_default,
           attribute.attstattarget::integer AS statistics_target,
           attribute.attoptions IS NULL AS options_are_empty,
           attribute.attfdwoptions IS NULL AS fdw_options_are_empty,
           attribute.attacl IS NULL AS acl_is_null
      FROM observed_relations AS relation
      JOIN pg_catalog.pg_attribute AS attribute
        ON attribute.attrelid = relation.oid
      JOIN pg_catalog.pg_type AS data_type ON data_type.oid = attribute.atttypid
      LEFT JOIN pg_catalog.pg_attrdef AS default_row
        ON default_row.adrelid = attribute.attrelid
       AND default_row.adnum = attribute.attnum
     WHERE attribute.attnum > 0
       AND NOT attribute.attisdropped
),
relation_constraints AS (
    SELECT constraint_row.conrelid AS relation_oid,
           constraint_row.oid,
           constraint_row.conname AS name,
           constraint_row.contype,
           constraint_row.convalidated,
           constraint_row.condeferrable,
           constraint_row.condeferred,
           constraint_row.connoinherit,
           constraint_row.conindid,
           constraint_row.conparentid,
           constraint_row.conkey,
           pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
               pg_catalog.pg_get_constraintdef(constraint_row.oid, true),
               'UTF8'
           )), 'hex') AS definition_sha256
      FROM observed_relations AS relation
      JOIN pg_catalog.pg_constraint AS constraint_row
        ON constraint_row.conrelid = relation.oid
       AND constraint_row.contype <> 'n'
),
relation_indexes AS (
    SELECT index_row.indrelid AS relation_oid,
           index_row.indexrelid AS index_oid,
           index_class.relname AS name,
           owner.oid AS owner_oid,
           owner.rolname AS owner,
           index_class.relkind,
           index_class.relpersistence,
           access_method.amname AS access_method,
           index_class.reltablespace,
           index_class.reloptions,
           index_row.indisunique,
           index_row.indnullsnotdistinct,
           index_row.indisprimary,
           index_row.indisexclusion,
           index_row.indimmediate,
           index_row.indisclustered,
           index_row.indisvalid,
           index_row.indcheckxmin,
           index_row.indisready,
           index_row.indislive,
           index_row.indisreplident,
           index_row.indnkeyatts,
           index_row.indnatts,
           index_row.indkey,
           index_row.indcollation,
           index_row.indclass,
           index_row.indoption,
           index_row.indexprs,
           index_row.indpred
      FROM observed_relations AS relation
      JOIN pg_catalog.pg_index AS index_row
        ON index_row.indrelid = relation.oid
      JOIN pg_catalog.pg_class AS index_class
        ON index_class.oid = index_row.indexrelid
      JOIN pg_catalog.pg_roles AS owner ON owner.oid = index_class.relowner
      LEFT JOIN pg_catalog.pg_am AS access_method
        ON access_method.oid = index_class.relam
),
relation_user_triggers AS (
    SELECT trigger_row.tgrelid AS relation_oid,
           trigger_row.oid,
           trigger_row.tgname AS name,
           trigger_row.tgfoid AS function_oid,
           routine.proname AS function_name,
           routine_namespace.nspname AS function_schema,
           trigger_row.tgconstraint AS constraint_oid,
           trigger_row.tgenabled,
           trigger_row.tgtype
      FROM observed_relations AS relation
      JOIN pg_catalog.pg_trigger AS trigger_row
        ON trigger_row.tgrelid = relation.oid
       AND NOT trigger_row.tgisinternal
      JOIN pg_catalog.pg_proc AS routine ON routine.oid = trigger_row.tgfoid
      JOIN pg_catalog.pg_namespace AS routine_namespace
        ON routine_namespace.oid = routine.pronamespace
),
relation_rules AS (
    SELECT rewrite_row.ev_class AS relation_oid,
           rewrite_row.oid,
           rewrite_row.rulename AS name,
           rewrite_row.ev_type,
           rewrite_row.ev_enabled,
           rewrite_row.is_instead
      FROM observed_relations AS relation
      JOIN pg_catalog.pg_rewrite AS rewrite_row
        ON rewrite_row.ev_class = relation.oid
),
relation_policies AS (
    SELECT policy_row.polrelid AS relation_oid,
           policy_row.oid,
           policy_row.polname AS name,
           policy_row.polpermissive,
           policy_row.polcmd,
           policy_row.polroles,
           policy_row.polqual,
           policy_row.polwithcheck
      FROM observed_relations AS relation
      JOIN pg_catalog.pg_policy AS policy_row
        ON policy_row.polrelid = relation.oid
),
relation_inheritance AS (
    SELECT relation.oid AS relation_oid,
           inheritance.inhrelid AS child_oid,
           inheritance.inhparent AS parent_oid,
           inheritance.inhseqno
      FROM observed_relations AS relation
      JOIN pg_catalog.pg_inherits AS inheritance
        ON inheritance.inhrelid = relation.oid
        OR inheritance.inhparent = relation.oid
),
relation_integrity AS (
    SELECT relation.label,
           relation.oid,
           pg_catalog.jsonb_build_object(
               'namespace_oid', relation.namespace_oid::text,
               'oid', relation.oid::text,
               'owner_oid', relation.owner_oid::text,
               'owner', relation.owner,
               'relation_kind', relation.relkind::text,
               'persistence', relation.relpersistence::text,
               'is_partition', relation.relispartition,
               'access_method', relation.access_method,
               'tablespace_oid', relation.reltablespace::text,
               'row_security', relation.relrowsecurity,
               'force_row_security', relation.relforcerowsecurity,
               'replica_identity', relation.relreplident::text,
               'options_are_empty', relation.reloptions IS NULL,
               'attribute_slots', relation.relnatts::integer,
               'relation_acl_is_null', relation.relacl IS NULL,
               'relation_acl', COALESCE((
                   SELECT pg_catalog.jsonb_agg(
                       pg_catalog.jsonb_build_object(
                           'grantor_oid', acl.grantor::text,
                           'grantor', acl.grantor_name,
                           'grantee_oid', acl.grantee::text,
                           'grantee', acl.grantee_name,
                           'privilege', acl.privilege_type,
                           'grantable', acl.is_grantable
                       ) ORDER BY acl.grantee_name, acl.privilege_type,
                                  acl.grantor_name
                   )
                     FROM relation_acl_rows AS acl
                    WHERE acl.relation_oid = relation.oid
               ), '[]'::jsonb),
               'columns', COALESCE((
                   SELECT pg_catalog.jsonb_agg(
                       pg_catalog.jsonb_build_object(
                           'position', column_row.position,
                           'name', column_row.name,
                           'type_oid', column_row.type_oid,
                           'type', column_row.type,
                           'not_null', column_row.not_null,
                           'has_default', column_row.has_default,
                           'default_expression_sha256',
                               column_row.default_expression_sha256,
                           'identity', column_row.identity,
                           'generated', column_row.generated,
                           'has_missing', column_row.has_missing,
                           'is_local', column_row.is_local,
                           'inheritance_count', column_row.inheritance_count,
                           'array_dimensions', column_row.array_dimensions,
                           'collation_is_type_default',
                               column_row.collation_is_type_default,
                           'storage_is_type_default',
                               column_row.storage_is_type_default,
                           'statistics_target', column_row.statistics_target,
                           'options_are_empty', column_row.options_are_empty,
                           'fdw_options_are_empty',
                               column_row.fdw_options_are_empty,
                           'acl_is_null', column_row.acl_is_null,
                           'acl', COALESCE((
                               SELECT pg_catalog.jsonb_agg(
                                   pg_catalog.jsonb_build_object(
                                       'grantor_oid', acl.grantor::text,
                                       'grantor', acl.grantor_name,
                                       'grantee_oid', acl.grantee::text,
                                       'grantee', acl.grantee_name,
                                       'privilege', acl.privilege_type,
                                       'grantable', acl.is_grantable
                                   ) ORDER BY acl.grantee_name,
                                              acl.privilege_type,
                                              acl.grantor_name
                               )
                                 FROM relation_column_acl_rows AS acl
                                WHERE acl.relation_oid = relation.oid
                                  AND acl.attnum = column_row.position
                           ), '[]'::jsonb)
                       ) ORDER BY column_row.position
                   )
                     FROM relation_columns AS column_row
                    WHERE column_row.relation_oid = relation.oid
               ), '[]'::jsonb),
               'constraints', COALESCE((
                   SELECT pg_catalog.jsonb_agg(
                       pg_catalog.jsonb_build_object(
                           'oid', constraint_row.oid::text,
                           'name', constraint_row.name,
                           'type', constraint_row.contype::text,
                           'validated', constraint_row.convalidated,
                           'deferrable', constraint_row.condeferrable,
                           'initially_deferred', constraint_row.condeferred,
                           'no_inherit', constraint_row.connoinherit,
                           'index_oid', constraint_row.conindid::text,
                           'parent_constraint_oid',
                               constraint_row.conparentid::text,
                           'column_numbers', COALESCE((
                               SELECT pg_catalog.jsonb_agg(
                                   key_part.attnum::integer
                                   ORDER BY key_part.ordinal
                               )
                                 FROM pg_catalog.unnest(constraint_row.conkey)
                                      WITH ORDINALITY
                                      AS key_part(attnum, ordinal)
                           ), '[]'::jsonb),
                           'column_names', COALESCE((
                               SELECT pg_catalog.jsonb_agg(
                                   attribute.attname ORDER BY key_part.ordinal
                               )
                                 FROM pg_catalog.unnest(constraint_row.conkey)
                                      WITH ORDINALITY
                                      AS key_part(attnum, ordinal)
                                 JOIN pg_catalog.pg_attribute AS attribute
                                   ON attribute.attrelid = relation.oid
                                  AND attribute.attnum = key_part.attnum
                           ), '[]'::jsonb),
                           'definition_sha256',
                               constraint_row.definition_sha256
                       ) ORDER BY constraint_row.oid
                   )
                     FROM relation_constraints AS constraint_row
                    WHERE constraint_row.relation_oid = relation.oid
               ), '[]'::jsonb),
               'indexes', COALESCE((
                   SELECT pg_catalog.jsonb_agg(
                       pg_catalog.jsonb_build_object(
                           'oid', index_row.index_oid::text,
                           'name', index_row.name,
                           'owner_oid', index_row.owner_oid::text,
                           'owner', index_row.owner,
                           'relation_kind', index_row.relkind::text,
                           'persistence', index_row.relpersistence::text,
                           'access_method', index_row.access_method,
                           'tablespace_oid', index_row.reltablespace::text,
                           'options_are_empty',
                               index_row.reloptions IS NULL,
                           'unique', index_row.indisunique,
                           'nulls_not_distinct',
                               index_row.indnullsnotdistinct,
                           'primary', index_row.indisprimary,
                           'exclusion', index_row.indisexclusion,
                           'immediate', index_row.indimmediate,
                           'clustered', index_row.indisclustered,
                           'valid', index_row.indisvalid,
                           'check_xmin', index_row.indcheckxmin,
                           'ready', index_row.indisready,
                           'live', index_row.indislive,
                           'replica_identity', index_row.indisreplident,
                           'key_attribute_count',
                               index_row.indnkeyatts::integer,
                           'attribute_count', index_row.indnatts::integer,
                           'key_columns', COALESCE((
                               SELECT pg_catalog.jsonb_agg(
                                   pg_catalog.jsonb_build_object(
                                       'position', key_part.ordinal::integer,
                                       'attribute_number',
                                           key_part.attnum::integer,
                                       'name', attribute.attname
                                   ) ORDER BY key_part.ordinal
                               )
                                 FROM pg_catalog.unnest(index_row.indkey)
                                      WITH ORDINALITY
                                      AS key_part(attnum, ordinal)
                                 LEFT JOIN pg_catalog.pg_attribute AS attribute
                                   ON attribute.attrelid = relation.oid
                                  AND attribute.attnum = key_part.attnum
                           ), '[]'::jsonb),
                           'operator_classes', COALESCE((
                               SELECT pg_catalog.jsonb_agg(
                                   pg_catalog.jsonb_build_object(
                                       'position', operator.ordinal::integer,
                                       'oid', operator.class_oid::text,
                                       'schema', class_namespace.nspname,
                                       'name', operator_class.opcname
                                   ) ORDER BY operator.ordinal
                               )
                                 FROM pg_catalog.unnest(index_row.indclass)
                                      WITH ORDINALITY
                                      AS operator(class_oid, ordinal)
                                 JOIN pg_catalog.pg_opclass AS operator_class
                                   ON operator_class.oid = operator.class_oid
                                 JOIN pg_catalog.pg_namespace AS class_namespace
                                   ON class_namespace.oid =
                                      operator_class.opcnamespace
                           ), '[]'::jsonb),
                           'collation_oids', COALESCE((
                               SELECT pg_catalog.jsonb_agg(
                                   collation_part.collation_oid::text
                                   ORDER BY collation_part.ordinal
                               )
                                 FROM pg_catalog.unnest(index_row.indcollation)
                                      WITH ORDINALITY
                                      AS collation_part(
                                          collation_oid, ordinal
                                      )
                           ), '[]'::jsonb),
                           'index_options', COALESCE((
                               SELECT pg_catalog.jsonb_agg(
                                   option.value::integer ORDER BY option.ordinal
                               )
                                 FROM pg_catalog.unnest(index_row.indoption)
                                      WITH ORDINALITY
                                      AS option(value, ordinal)
                           ), '[]'::jsonb),
                           'expressions_present',
                               index_row.indexprs IS NOT NULL,
                           'expressions_sha256', CASE
                               WHEN index_row.indexprs IS NULL THEN NULL ELSE
                               pg_catalog.encode(pg_catalog.sha256(
                                   pg_catalog.convert_to(pg_catalog.pg_get_expr(
                                       index_row.indexprs,
                                       index_row.relation_oid,
                                       true
                                   ), 'UTF8')
                               ), 'hex') END,
                           'predicate_present',
                               index_row.indpred IS NOT NULL,
                           'predicate_sha256', CASE
                               WHEN index_row.indpred IS NULL THEN NULL ELSE
                               pg_catalog.encode(pg_catalog.sha256(
                                   pg_catalog.convert_to(pg_catalog.pg_get_expr(
                                       index_row.indpred,
                                       index_row.relation_oid,
                                       true
                                   ), 'UTF8')
                               ), 'hex') END,
                           'constraint_oids', COALESCE((
                               SELECT pg_catalog.jsonb_agg(
                                   constraint_row.oid::text
                                   ORDER BY constraint_row.oid
                               )
                                 FROM relation_constraints AS constraint_row
                                WHERE constraint_row.relation_oid = relation.oid
                                  AND constraint_row.conindid =
                                      index_row.index_oid
                           ), '[]'::jsonb)
                       ) ORDER BY index_row.index_oid
                   )
                     FROM relation_indexes AS index_row
                    WHERE index_row.relation_oid = relation.oid
               ), '[]'::jsonb),
               'user_triggers', COALESCE((
                   SELECT pg_catalog.jsonb_agg(
                       pg_catalog.jsonb_build_object(
                           'oid', trigger_row.oid::text,
                           'name', trigger_row.name,
                           'function_oid', trigger_row.function_oid::text,
                           'function_schema', trigger_row.function_schema,
                           'function_name', trigger_row.function_name,
                           'constraint_oid',
                               trigger_row.constraint_oid::text,
                           'enabled', trigger_row.tgenabled::text,
                           'type', trigger_row.tgtype::integer
                       ) ORDER BY trigger_row.oid
                   )
                     FROM relation_user_triggers AS trigger_row
                    WHERE trigger_row.relation_oid = relation.oid
               ), '[]'::jsonb),
               'rules', COALESCE((
                   SELECT pg_catalog.jsonb_agg(
                       pg_catalog.jsonb_build_object(
                           'oid', rule_row.oid::text,
                           'name', rule_row.name,
                           'event_type', rule_row.ev_type::text,
                           'enabled', rule_row.ev_enabled::text,
                           'instead', rule_row.is_instead
                       ) ORDER BY rule_row.oid
                   )
                     FROM relation_rules AS rule_row
                    WHERE rule_row.relation_oid = relation.oid
               ), '[]'::jsonb),
               'policies', COALESCE((
                   SELECT pg_catalog.jsonb_agg(
                       pg_catalog.jsonb_build_object(
                           'oid', policy_row.oid::text,
                           'name', policy_row.name,
                           'permissive', policy_row.polpermissive,
                           'command', policy_row.polcmd::text,
                           'role_oids', COALESCE((
                               SELECT pg_catalog.jsonb_agg(
                                   role_oid::text ORDER BY role_oid
                               ) FROM pg_catalog.unnest(policy_row.polroles)
                                    AS role_oid
                           ), '[]'::jsonb),
                           'using_expression_present',
                               policy_row.polqual IS NOT NULL,
                           'with_check_expression_present',
                               policy_row.polwithcheck IS NOT NULL
                       ) ORDER BY policy_row.oid
                   )
                     FROM relation_policies AS policy_row
                    WHERE policy_row.relation_oid = relation.oid
               ), '[]'::jsonb),
               'inheritance', COALESCE((
                   SELECT pg_catalog.jsonb_agg(
                       pg_catalog.jsonb_build_object(
                           'child_oid', inheritance.child_oid::text,
                           'parent_oid', inheritance.parent_oid::text,
                           'sequence', inheritance.inhseqno::integer
                       ) ORDER BY inheritance.child_oid,
                                  inheritance.parent_oid,
                                  inheritance.inhseqno
                   )
                     FROM relation_inheritance AS inheritance
                    WHERE inheritance.relation_oid = relation.oid
               ), '[]'::jsonb)
           ) AS value
      FROM observed_relations AS relation
),
event_log_observation AS (
    SELECT pg_catalog.jsonb_build_object(
        'cardinality', (
            SELECT pg_catalog.count(*) FROM observed_relations
             WHERE label = 'event_log'
        ),
        'identity', (
            SELECT value FROM relation_integrity WHERE label = 'event_log'
        )
    ) AS value
),
legacy_archive_observation AS (
    SELECT pg_catalog.jsonb_build_object(
        'cardinality', (
            SELECT pg_catalog.count(*) FROM observed_relations
             WHERE label = 'legacy_archive'
        ),
        'identity', (
            SELECT integrity.value || pg_catalog.jsonb_build_object(
                'owner_superuser', relation.owner_superuser,
                'owner_create_database', relation.owner_create_database,
                'owner_create_role', relation.owner_create_role,
                'owner_replication', relation.owner_replication,
                'owner_bypass_row_security',
                    relation.owner_bypass_row_security,
                'owner_connection_limit', relation.owner_connection_limit,
                'owner_validity_is_unbounded',
                    relation.owner_validity_is_unbounded,
                'owner_configuration_is_empty',
                    relation.owner_configuration_is_empty
            )
              FROM relation_integrity AS integrity
              JOIN observed_relations AS relation
                ON relation.oid = integrity.oid
             WHERE integrity.label = 'legacy_archive'
        )
    ) AS value
),
writer_ping_routines AS (
    SELECT routine.oid,
           namespace.oid AS namespace_oid,
           owner.oid AS owner_oid,
           owner.rolname AS owner,
           language.lanname AS language,
           routine.prokind,
           routine.proretset,
           return_namespace.nspname AS return_type_schema,
           return_type.typname AS return_type_name,
           routine.prosecdef,
           routine.proleakproof,
           routine.proisstrict,
           routine.provolatile,
           routine.proparallel,
           routine.proconfig,
           routine.proacl,
           routine.proowner,
           routine.proargtypes,
           pg_catalog.encode(
               pg_catalog.sha256(pg_catalog.convert_to(
                   pg_catalog.jsonb_build_object(
                       'source', routine.prosrc,
                       'binary', routine.probin,
                       'sql_body', pg_catalog.to_jsonb(routine.prosqlbody)
                   )::text,
                   'UTF8'
               )),
               'hex'
           ) AS implementation_sha256
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
      JOIN pg_catalog.pg_roles AS owner ON owner.oid = routine.proowner
      JOIN pg_catalog.pg_language AS language ON language.oid = routine.prolang
      JOIN pg_catalog.pg_type AS return_type ON return_type.oid = routine.prorettype
      JOIN pg_catalog.pg_namespace AS return_namespace
        ON return_namespace.oid = return_type.typnamespace
     WHERE namespace.nspname = 'canonical_brain'
       AND routine.proname = 'writer_ping'
),
writer_ping_observation AS (
    SELECT pg_catalog.jsonb_build_object(
        'cardinality', (SELECT pg_catalog.count(*) FROM writer_ping_routines),
        'routines', COALESCE((
            SELECT pg_catalog.jsonb_agg(
                pg_catalog.jsonb_build_object(
                    'oid', routine.oid::text,
                    'namespace_oid', routine.namespace_oid::text,
                    'owner_oid', routine.owner_oid::text,
                    'owner', routine.owner,
                    'language', routine.language,
                    'kind', routine.prokind::text,
                    'argument_types', COALESCE((
                        SELECT pg_catalog.jsonb_agg(
                            pg_catalog.jsonb_build_object(
                                'position', argument.position::integer,
                                'schema', type_namespace.nspname,
                                'name', argument_type.typname
                            ) ORDER BY argument.position
                        )
                          FROM pg_catalog.unnest(routine.proargtypes)
                               WITH ORDINALITY AS argument(type_oid, position)
                          JOIN pg_catalog.pg_type AS argument_type
                            ON argument_type.oid = argument.type_oid
                          JOIN pg_catalog.pg_namespace AS type_namespace
                            ON type_namespace.oid = argument_type.typnamespace
                    ), '[]'::jsonb),
                    'return_type', pg_catalog.jsonb_build_object(
                        'schema', routine.return_type_schema,
                        'name', routine.return_type_name
                    ),
                    'returns_set', routine.proretset,
                    'security_definer', routine.prosecdef,
                    'leakproof', routine.proleakproof,
                    'strict', routine.proisstrict,
                    'volatility', routine.provolatile::text,
                    'parallel', routine.proparallel::text,
                    'configuration_count', COALESCE(
                        pg_catalog.cardinality(routine.proconfig), 0
                    ),
                    'configuration_is_exact',
                        routine.proconfig = ARRAY[
                            'search_path=pg_catalog, canonical_brain'
                        ]::text[],
                    'acl_is_null', routine.proacl IS NULL,
                    'acl', COALESCE((
                        SELECT pg_catalog.jsonb_agg(
                            pg_catalog.jsonb_build_object(
                                'grantor_oid', acl.grantor::text,
                                'grantor', acl_grantor.rolname,
                                'grantee_oid', acl.grantee::text,
                                'grantee', CASE WHEN acl.grantee = 0
                                    THEN 'PUBLIC' ELSE acl_grantee.rolname END,
                                'privilege', acl.privilege_type,
                                'grantable', acl.is_grantable
                            ) ORDER BY
                                CASE WHEN acl.grantee = 0
                                    THEN 'PUBLIC' ELSE acl_grantee.rolname END,
                                acl.privilege_type,
                                acl_grantor.rolname
                        )
                          FROM pg_catalog.aclexplode(COALESCE(
                              routine.proacl,
                              pg_catalog.acldefault('f', routine.proowner)
                          )) AS acl
                          JOIN pg_catalog.pg_roles AS acl_grantor
                            ON acl_grantor.oid = acl.grantor
                          LEFT JOIN pg_catalog.pg_roles AS acl_grantee
                            ON acl_grantee.oid = acl.grantee
                    ), '[]'::jsonb),
                    'implementation_sha256', routine.implementation_sha256
                ) ORDER BY routine.oid
            ) FROM writer_ping_routines AS routine
        ), '[]'::jsonb)
    ) AS value
),
database_scope AS (
    SELECT database.oid,
           database.datname AS name,
           owner.oid AS owner_oid,
           owner.rolname AS owner,
           database.datallowconn AS allow_connections,
           database.datistemplate AS is_template,
           database.datconnlimit AS connection_limit,
           database.datacl,
           database.datdba
      FROM pg_catalog.pg_database AS database
      JOIN pg_catalog.pg_roles AS owner ON owner.oid = database.datdba
     WHERE database.datallowconn
        OR database.datname = pg_catalog.current_database()
),
database_acl_rows AS (
    SELECT database.oid AS database_oid,
           acl.grantor,
           grantor.rolname AS grantor_name,
           acl.grantee,
           CASE WHEN acl.grantee = 0 THEN 'PUBLIC'
                ELSE grantee.rolname END AS grantee_name,
           acl.privilege_type,
           acl.is_grantable
      FROM database_scope AS database
      CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(
          database.datacl,
          pg_catalog.acldefault('d', database.datdba)
      )) AS acl
      JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = acl.grantor
      LEFT JOIN pg_catalog.pg_roles AS grantee ON grantee.oid = acl.grantee
),
database_observations AS (
    SELECT database.oid,
           database.name,
           database.allow_connections,
           pg_catalog.jsonb_build_object(
               'oid', database.oid::text,
               'name', database.name,
               'owner_oid', database.owner_oid::text,
               'owner', database.owner,
               'allow_connections', database.allow_connections,
               'is_template', database.is_template,
               'connection_limit', database.connection_limit,
               'acl_is_null', database.datacl IS NULL,
               'acl', COALESCE((
                   SELECT pg_catalog.jsonb_agg(
                       pg_catalog.jsonb_build_object(
                           'grantor_oid', acl.grantor::text,
                           'grantor', acl.grantor_name,
                           'grantee_oid', acl.grantee::text,
                           'grantee', acl.grantee_name,
                           'privilege', acl.privilege_type,
                           'grantable', acl.is_grantable
                       ) ORDER BY acl.grantee_name, acl.privilege_type,
                                  acl.grantor_name
                   )
                     FROM database_acl_rows AS acl
                    WHERE acl.database_oid = database.oid
               ), '[]'::jsonb),
               'effective_public_connect', COALESCE((
                   SELECT pg_catalog.bool_or(
                       acl.grantee = 0 AND acl.privilege_type = 'CONNECT'
                   )
                     FROM database_acl_rows AS acl
                    WHERE acl.database_oid = database.oid
               ), false),
               'effective_public_temporary', COALESCE((
                   SELECT pg_catalog.bool_or(
                       acl.grantee = 0 AND acl.privilege_type = 'TEMPORARY'
                   )
                     FROM database_acl_rows AS acl
                    WHERE acl.database_oid = database.oid
               ), false)
           ) AS value
      FROM database_scope AS database
),
managed_cloudsqladmin_role AS (
    SELECT role.oid,
           pg_catalog.jsonb_build_object(
               'oid', role.oid::text,
               'name', role.rolname,
               'can_login', role.rolcanlogin,
               'inherits', role.rolinherit,
               'superuser', role.rolsuper,
               'create_database', role.rolcreatedb,
               'create_role', role.rolcreaterole,
               'replication', role.rolreplication,
               'bypass_row_security', role.rolbypassrls,
               'connection_limit', role.rolconnlimit,
               'validity_is_unbounded', role.rolvaliduntil IS NULL,
               'configuration_is_empty', role.rolconfig IS NULL
           ) AS value
      FROM pg_catalog.pg_roles AS role
     WHERE role.rolname = 'cloudsqladmin'
),
managed_cloudsqladmin_observation AS (
    SELECT pg_catalog.jsonb_build_object(
        'role_cardinality', (
            SELECT pg_catalog.count(*) FROM managed_cloudsqladmin_role
        ),
        'role', (SELECT value FROM managed_cloudsqladmin_role),
        'database_privileges', COALESCE((
            SELECT pg_catalog.jsonb_agg(
                pg_catalog.jsonb_build_object(
                    'database_oid', database.oid::text,
                    'database', database.name,
                    'effective_connect', pg_catalog.has_database_privilege(
                        managed.oid, database.oid, 'CONNECT'
                    ),
                    'effective_temporary', pg_catalog.has_database_privilege(
                        managed.oid, database.oid, 'TEMPORARY'
                    ),
                    'direct_acl', COALESCE((
                        SELECT pg_catalog.jsonb_agg(
                            pg_catalog.jsonb_build_object(
                                'grantor_oid', acl.grantor::text,
                                'grantor', acl.grantor_name,
                                'privilege', acl.privilege_type,
                                'grantable', acl.is_grantable
                            ) ORDER BY acl.privilege_type, acl.grantor_name
                        )
                          FROM database_acl_rows AS acl
                         WHERE acl.database_oid = database.oid
                           AND acl.grantee = managed.oid
                    ), '[]'::jsonb)
                ) ORDER BY database.name
            )
              FROM managed_cloudsqladmin_role AS managed
              CROSS JOIN database_scope AS database
        ), '[]'::jsonb)
    ) AS value
),
receipt AS (
    SELECT pg_catalog.jsonb_build_object(
        'schema',
            'muncho-canonical-writer-foundation-phase-b-db-preflight.v1',
        'preflight', true,
        'terminal', false,
        'database', pg_catalog.current_database(),
        'database_owner', (
            SELECT database.owner
              FROM database_scope AS database
             WHERE database.name = pg_catalog.current_database()
        ),
        'postgres_version_num',
            pg_catalog.current_setting('server_version_num')::integer,
        'session_user', SESSION_USER,
        'current_user', CURRENT_USER,
        'roles', COALESCE((
            SELECT pg_catalog.jsonb_agg(
                pg_catalog.to_jsonb(role_row) ORDER BY role_row.name
            ) FROM target_roles AS role_row
        ), '[]'::jsonb),
        'memberships', COALESCE((
            SELECT pg_catalog.jsonb_agg(
                pg_catalog.to_jsonb(membership_row)
                ORDER BY membership_row.granted_role,
                         membership_row.member_role,
                         membership_row.grantor
            ) FROM observed_memberships AS membership_row
        ), '[]'::jsonb),
        'temporary_admin_roles', COALESCE((
            SELECT pg_catalog.jsonb_agg(
                pg_catalog.to_jsonb(role_row) ORDER BY role_row.name
            ) FROM temporary_admin_roles AS role_row
        ), '[]'::jsonb),
        'bootstrap_role_absent', NOT EXISTS (
            SELECT 1 FROM target_roles
             WHERE name = 'canonical_brain_canary_bootstrap'
        ),
        'bootstrap_login_absent', NOT EXISTS (
            SELECT 1 FROM target_roles
             WHERE name = 'canonical_brain_canary_bootstrap_login'
        ),
        'namespaces', COALESCE((
            SELECT pg_catalog.jsonb_agg(
                namespace.value ORDER BY namespace.name
            ) FROM namespace_observations AS namespace
        ), '[]'::jsonb),
        'event_log', (SELECT value FROM event_log_observation),
        'writer_ping', (SELECT value FROM writer_ping_observation),
        'legacy_archive', (SELECT value FROM legacy_archive_observation),
        'target_database', (
            SELECT database.value
              FROM database_observations AS database
             WHERE database.name = pg_catalog.current_database()
        ),
        'other_connectable_databases', COALESCE((
            SELECT pg_catalog.jsonb_agg(
                database.value ORDER BY database.name
            )
              FROM database_observations AS database
             WHERE database.allow_connections
               AND database.name <> pg_catalog.current_database()
        ), '[]'::jsonb),
        'managed_cloudsqladmin', (
            SELECT value FROM managed_cloudsqladmin_observation
        ),
        'secret_material_recorded', false
    ) AS value
),
receipt_payload AS (
    SELECT receipt.value,
           receipt.value::text AS unsigned_receipt_jsonb_text
      FROM receipt
)
SELECT (
    receipt.value || pg_catalog.jsonb_build_object(
        'unsigned_receipt_jsonb_text',
            receipt.unsigned_receipt_jsonb_text,
        'receipt_sha256', pg_catalog.encode(
            pg_catalog.sha256(
                pg_catalog.convert_to(
                    receipt.unsigned_receipt_jsonb_text, 'UTF8'
                )
            ),
            'hex'
        )
    )
)::text AS phase_b_database_preflight
FROM receipt_payload AS receipt;

COMMIT;
