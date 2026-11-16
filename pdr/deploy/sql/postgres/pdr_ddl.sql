-- 由 server/pdr_server/db/tables.py 生成，请勿手改：python -m pdr_server.db.render
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS src;
CREATE SCHEMA IF NOT EXISTS ods;
CREATE SCHEMA IF NOT EXISTS dwd;
CREATE SCHEMA IF NOT EXISTS dws;
CREATE SCHEMA IF NOT EXISTS ads;
CREATE SCHEMA IF NOT EXISTS dim;

-- ======================================================================
-- src
-- ======================================================================

CREATE TABLE IF NOT EXISTS src.src_cus_tenant_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  tenant_name varchar(128) NOT NULL,
  tenant_type smallint NOT NULL,
  tenant_status smallint NOT NULL,
  agent_profile_name varchar(64) NOT NULL,
  expire_date date,
  max_user_qty int,
  max_storage_gb_qty int,
  contact_name varchar(64),
  contact_phone_no varchar(32),
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code)
);
COMMENT ON TABLE src.src_cus_tenant_f IS '租户表（SaaS 租户主数据）';
COMMENT ON COLUMN src.src_cus_tenant_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_tenant_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_tenant_f.tenant_name IS '租户名称';
COMMENT ON COLUMN src.src_cus_tenant_f.tenant_type IS '租户类型：0-平台运营方 1-企业租户 2-试用租户';
COMMENT ON COLUMN src.src_cus_tenant_f.tenant_status IS '租户状态：0-正常 1-停用 2-已过期';
COMMENT ON COLUMN src.src_cus_tenant_f.agent_profile_name IS '对应 hermes 智能体 profile 名称，形如 t_<tenant_code>';
COMMENT ON COLUMN src.src_cus_tenant_f.expire_date IS '服务到期日期';
COMMENT ON COLUMN src.src_cus_tenant_f.max_user_qty IS '最大用户数配额';
COMMENT ON COLUMN src.src_cus_tenant_f.max_storage_gb_qty IS '存储配额（GB）';
COMMENT ON COLUMN src.src_cus_tenant_f.contact_name IS '联系人姓名';
COMMENT ON COLUMN src.src_cus_tenant_f.contact_phone_no IS '联系人电话号码';
COMMENT ON COLUMN src.src_cus_tenant_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_tenant_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_tenant_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_tenant_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_tenant_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_tenant_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_tenant_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_tenant_f;
CREATE POLICY p_tenant_read ON src.src_cus_tenant_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_tenant_f;
CREATE POLICY p_tenant_write ON src.src_cus_tenant_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_cus_user_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  user_code varchar(64) NOT NULL,
  user_name varchar(64) NOT NULL,
  password_hash varchar(256) NOT NULL,
  email_addr varchar(128),
  mobile_no varchar(32),
  user_type smallint NOT NULL,
  user_status smallint NOT NULL,
  last_login_time timestamp(3),
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, user_code)
);
COMMENT ON TABLE src.src_cus_user_f IS '用户表';
COMMENT ON COLUMN src.src_cus_user_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_user_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_user_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_cus_user_f.user_code IS '用户编码（租户内唯一，登录名）';
COMMENT ON COLUMN src.src_cus_user_f.user_name IS '用户姓名';
COMMENT ON COLUMN src.src_cus_user_f.password_hash IS '密码哈希（argon2id）';
COMMENT ON COLUMN src.src_cus_user_f.email_addr IS '电子邮箱地址';
COMMENT ON COLUMN src.src_cus_user_f.mobile_no IS '手机号码';
COMMENT ON COLUMN src.src_cus_user_f.user_type IS '用户类型：0-平台超级管理员 1-租户管理员 2-管理端普通用户 3-API 账号 4-用户端问答用户（只能登录用户端，0~2 只能登录管理端）';
COMMENT ON COLUMN src.src_cus_user_f.user_status IS '用户状态：0-正常 1-锁定 2-停用';
COMMENT ON COLUMN src.src_cus_user_f.last_login_time IS '最近登录时间';
COMMENT ON COLUMN src.src_cus_user_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_user_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_user_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_user_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_user_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_user_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_user_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_user_f;
CREATE POLICY p_tenant_read ON src.src_cus_user_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_user_f;
CREATE POLICY p_tenant_write ON src.src_cus_user_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_cus_role_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  role_code varchar(64) NOT NULL,
  role_name varchar(64) NOT NULL,
  role_type smallint NOT NULL,
  role_desc varchar(512),
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, role_code)
);
COMMENT ON TABLE src.src_cus_role_f IS '角色表';
COMMENT ON COLUMN src.src_cus_role_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_role_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_role_f.role_code IS '角色编码';
COMMENT ON COLUMN src.src_cus_role_f.role_name IS '角色名称';
COMMENT ON COLUMN src.src_cus_role_f.role_type IS '角色类型：0-系统内置（不可删） 1-租户自定义';
COMMENT ON COLUMN src.src_cus_role_f.role_desc IS '角色说明';
COMMENT ON COLUMN src.src_cus_role_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_role_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_role_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_role_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_role_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_role_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_role_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_role_f;
CREATE POLICY p_tenant_read ON src.src_cus_role_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_role_f;
CREATE POLICY p_tenant_write ON src.src_cus_role_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_cus_permission_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  permission_code varchar(128) NOT NULL,
  permission_name varchar(128) NOT NULL,
  client_type varchar(16) NOT NULL,
  permission_type smallint NOT NULL,
  parent_permission_code varchar(128) NOT NULL DEFAULT '',
  api_method_name varchar(16),
  api_path_name varchar(256),
  route_path_name varchar(256),
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, permission_code)
);
COMMENT ON TABLE src.src_cus_permission_f IS '权限点表（菜单/按钮/接口，全平台共享，tenant_code=platform）';
COMMENT ON COLUMN src.src_cus_permission_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_permission_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_permission_f.permission_code IS '权限点编码，形如 content:review:submit';
COMMENT ON COLUMN src.src_cus_permission_f.permission_name IS '权限点名称';
COMMENT ON COLUMN src.src_cus_permission_f.client_type IS '所属端：admin-管理端 portal-用户端（两端权限点互不相通）';
COMMENT ON COLUMN src.src_cus_permission_f.permission_type IS '权限点类型：1-菜单 2-按钮 3-接口';
COMMENT ON COLUMN src.src_cus_permission_f.parent_permission_code IS '上级权限点编码';
COMMENT ON COLUMN src.src_cus_permission_f.api_method_name IS '接口 HTTP 方法（类型为接口时）：GET/POST/PUT/PATCH/DELETE';
COMMENT ON COLUMN src.src_cus_permission_f.api_path_name IS '接口路径模板（类型为接口时），如 /api/content/assets/{id}';
COMMENT ON COLUMN src.src_cus_permission_f.route_path_name IS '前端路由路径（类型为菜单时）';
COMMENT ON COLUMN src.src_cus_permission_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_permission_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_permission_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_permission_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_permission_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_permission_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_permission_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_permission_f;
CREATE POLICY p_tenant_read ON src.src_cus_permission_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_permission_f;
CREATE POLICY p_tenant_write ON src.src_cus_permission_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_cus_user_role_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  user_code varchar(64) NOT NULL,
  role_code varchar(64) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, user_code, role_code)
);
COMMENT ON TABLE src.src_cus_user_role_f IS '用户-角色关系表';
COMMENT ON COLUMN src.src_cus_user_role_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_user_role_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_user_role_f.user_code IS '用户编码';
COMMENT ON COLUMN src.src_cus_user_role_f.role_code IS '角色编码';
COMMENT ON COLUMN src.src_cus_user_role_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_user_role_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_user_role_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_user_role_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_user_role_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_user_role_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_user_role_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_user_role_f;
CREATE POLICY p_tenant_read ON src.src_cus_user_role_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_user_role_f;
CREATE POLICY p_tenant_write ON src.src_cus_user_role_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_cus_role_permission_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  role_code varchar(64) NOT NULL,
  permission_code varchar(128) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, role_code, permission_code)
);
COMMENT ON TABLE src.src_cus_role_permission_f IS '角色-权限点关系表';
COMMENT ON COLUMN src.src_cus_role_permission_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_role_permission_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_role_permission_f.role_code IS '角色编码';
COMMENT ON COLUMN src.src_cus_role_permission_f.permission_code IS '权限点编码';
COMMENT ON COLUMN src.src_cus_role_permission_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_role_permission_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_role_permission_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_role_permission_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_role_permission_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_role_permission_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_role_permission_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_role_permission_f;
CREATE POLICY p_tenant_read ON src.src_cus_role_permission_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_role_permission_f;
CREATE POLICY p_tenant_write ON src.src_cus_role_permission_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_cus_user_region_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  user_code varchar(64) NOT NULL,
  primary_status smallint NOT NULL DEFAULT 0,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, user_code, region_code)
);
COMMENT ON TABLE src.src_cus_user_region_f IS '用户归属区域表（用户可归属多个区域，作为“本区域”取值来源）';
COMMENT ON COLUMN src.src_cus_user_region_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_user_region_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_user_region_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_cus_user_region_f.user_code IS '用户编码';
COMMENT ON COLUMN src.src_cus_user_region_f.primary_status IS '是否主区域：0-否 1-是';
COMMENT ON COLUMN src.src_cus_user_region_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_user_region_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_user_region_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_user_region_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_user_region_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_user_region_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_user_region_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_user_region_f;
CREATE POLICY p_tenant_read ON src.src_cus_user_region_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_user_region_f;
CREATE POLICY p_tenant_write ON src.src_cus_user_region_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_cus_role_data_scope_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  role_code varchar(64) NOT NULL,
  resource_type varchar(64) NOT NULL,
  scope_type smallint NOT NULL,
  action_type varchar(16) NOT NULL DEFAULT '*',
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, role_code, resource_type, action_type)
);
COMMENT ON TABLE src.src_cus_role_data_scope_f IS '角色数据范围表（行级权限：按资源类型配置可见范围）';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.role_code IS '角色编码';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.resource_type IS '资源类型：content_asset-内容资产 knowledge-知识库 business_extract-业务回填数据 report-报表 *-全部';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.scope_type IS '范围类型：1-本租户全部 2-本区域及下级 3-仅本区域 4-自定义区域 5-仅本人创建';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.action_type IS '适用操作：read-读 write-写 *-读写';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_role_data_scope_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_role_data_scope_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_role_data_scope_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_role_data_scope_f;
CREATE POLICY p_tenant_read ON src.src_cus_role_data_scope_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_role_data_scope_f;
CREATE POLICY p_tenant_write ON src.src_cus_role_data_scope_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_cus_role_region_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  role_code varchar(64) NOT NULL,
  resource_type varchar(64) NOT NULL,
  include_child_status smallint NOT NULL DEFAULT 1,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, role_code, resource_type, region_code)
);
COMMENT ON TABLE src.src_cus_role_region_f IS '角色自定义区域表（数据范围为 4-自定义区域 时生效）';
COMMENT ON COLUMN src.src_cus_role_region_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_role_region_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_role_region_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_cus_role_region_f.role_code IS '角色编码';
COMMENT ON COLUMN src.src_cus_role_region_f.resource_type IS '资源类型：content_asset-内容资产 knowledge-知识库 business_extract-业务回填数据 report-报表 *-全部';
COMMENT ON COLUMN src.src_cus_role_region_f.include_child_status IS '是否包含下级区域：0-否 1-是';
COMMENT ON COLUMN src.src_cus_role_region_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_role_region_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_role_region_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_role_region_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_role_region_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_role_region_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_role_region_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_role_region_f;
CREATE POLICY p_tenant_read ON src.src_cus_role_region_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_role_region_f;
CREATE POLICY p_tenant_write ON src.src_cus_role_region_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_cus_role_column_perm_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  role_code varchar(64) NOT NULL,
  table_name varchar(128) NOT NULL,
  column_name varchar(128) NOT NULL,
  perm_type smallint NOT NULL,
  mask_rule_type varchar(32),
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, role_code, table_name, column_name)
);
COMMENT ON TABLE src.src_cus_role_column_perm_f IS '角色列级权限表（未配置的列默认可读写）';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.role_code IS '角色编码';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.table_name IS '受控表（库名.表名），如 ods.ods_fin_invoice_extract_di';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.column_name IS '受控字段名';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.perm_type IS '列权限：0-不可见 1-脱敏可见 2-只读 3-可读写';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.mask_rule_type IS '脱敏规则：mobile-手机号 id_card-证件号 amount-金额区间 name-姓名 full-全遮蔽';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_role_column_perm_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_role_column_perm_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_role_column_perm_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_role_column_perm_f;
CREATE POLICY p_tenant_read ON src.src_cus_role_column_perm_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_role_column_perm_f;
CREATE POLICY p_tenant_write ON src.src_cus_role_column_perm_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_cus_api_token_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  user_code varchar(64) NOT NULL,
  token_name varchar(64) NOT NULL,
  token_hash varchar(128) NOT NULL,
  token_prefix_code varchar(16) NOT NULL,
  expire_time timestamp(3),
  token_status smallint NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (token_hash)
);
COMMENT ON TABLE src.src_cus_api_token_f IS 'API 访问令牌表（供 API 账号与外部系统调用）';
COMMENT ON COLUMN src.src_cus_api_token_f.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_api_token_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_api_token_f.user_code IS '所属用户编码';
COMMENT ON COLUMN src.src_cus_api_token_f.token_name IS '令牌名称';
COMMENT ON COLUMN src.src_cus_api_token_f.token_hash IS '令牌 SHA-256 哈希（明文只在创建时展示一次）';
COMMENT ON COLUMN src.src_cus_api_token_f.token_prefix_code IS '令牌前缀（用于界面识别）';
COMMENT ON COLUMN src.src_cus_api_token_f.expire_time IS '过期时间，为空表示不过期';
COMMENT ON COLUMN src.src_cus_api_token_f.token_status IS '令牌状态：0-有效 1-已吊销';
COMMENT ON COLUMN src.src_cus_api_token_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_api_token_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_api_token_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_api_token_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_api_token_f.update_time IS '最后更新时间';
ALTER TABLE src.src_cus_api_token_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_api_token_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_api_token_f;
CREATE POLICY p_tenant_read ON src.src_cus_api_token_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_api_token_f;
CREATE POLICY p_tenant_write ON src.src_cus_api_token_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_cus_operate_log_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  user_code varchar(64) NOT NULL,
  client_type varchar(16) NOT NULL,
  operate_type varchar(64) NOT NULL,
  resource_type varchar(64),
  resource_no varchar(128),
  request_path_name varchar(256),
  client_ip_code varchar(64),
  operate_status smallint NOT NULL,
  detail_json jsonb,
  operate_time timestamp(3) NOT NULL,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS src.src_cus_operate_log_di_pdefault PARTITION OF src.src_cus_operate_log_di DEFAULT;
COMMENT ON TABLE src.src_cus_operate_log_di IS '操作审计日志（天增量）';
COMMENT ON COLUMN src.src_cus_operate_log_di.id IS '自增主键';
COMMENT ON COLUMN src.src_cus_operate_log_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_cus_operate_log_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_cus_operate_log_di.user_code IS '操作人用户编码';
COMMENT ON COLUMN src.src_cus_operate_log_di.client_type IS '请求来源端：admin-管理端 portal-用户端 api-API 令牌';
COMMENT ON COLUMN src.src_cus_operate_log_di.operate_type IS '操作类型，开放取值，与权限点编码一致，如 config:update / content:review:submit / role:grant';
COMMENT ON COLUMN src.src_cus_operate_log_di.resource_type IS '操作对象资源类型：content_asset-内容资产 knowledge-知识库 config-配置 user-用户 role-角色 tenant-租户 business_extract-业务回填数据';
COMMENT ON COLUMN src.src_cus_operate_log_di.resource_no IS '操作对象编号';
COMMENT ON COLUMN src.src_cus_operate_log_di.request_path_name IS '请求接口路径';
COMMENT ON COLUMN src.src_cus_operate_log_di.client_ip_code IS '客户端 IP';
COMMENT ON COLUMN src.src_cus_operate_log_di.operate_status IS '操作结果：0-成功 1-失败 2-拒绝（无权限）';
COMMENT ON COLUMN src.src_cus_operate_log_di.detail_json IS '操作详情（变更前后值等，敏感字段已脱敏）';
COMMENT ON COLUMN src.src_cus_operate_log_di.operate_time IS '操作时间';
COMMENT ON COLUMN src.src_cus_operate_log_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN src.src_cus_operate_log_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_cus_operate_log_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_cus_operate_log_di.create_time IS '创建时间';
COMMENT ON COLUMN src.src_cus_operate_log_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_cus_operate_log_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_src_cus_operate_log_di_0 ON src.src_cus_operate_log_di (tenant_code, user_code, operate_time);
ALTER TABLE src.src_cus_operate_log_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_cus_operate_log_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_cus_operate_log_di;
CREATE POLICY p_tenant_read ON src.src_cus_operate_log_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_cus_operate_log_di;
CREATE POLICY p_tenant_write ON src.src_cus_operate_log_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_config_item_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  config_key varchar(256) NOT NULL,
  config_domain_type varchar(16) NOT NULL,
  group_code varchar(64) NOT NULL,
  group_name varchar(64) NOT NULL,
  config_name varchar(128) NOT NULL,
  config_desc text,
  value_type varchar(16) NOT NULL,
  default_value jsonb,
  value_range jsonb,
  reload_type smallint NOT NULL,
  scope_type smallint NOT NULL,
  sort_qty int,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, config_key)
);
COMMENT ON TABLE src.src_opr_config_item_f IS '配置项字典（hermes 智能体与 PDR 系统配置项的中文元数据）';
COMMENT ON COLUMN src.src_opr_config_item_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_config_item_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_config_item_f.config_key IS '配置键（点分路径），如 hermes.agent.max_turns / pdr.pipeline.retry.max_attempt_qty';
COMMENT ON COLUMN src.src_opr_config_item_f.config_domain_type IS '配置归属：pdr-PDR系统配置 hermes-智能体配置';
COMMENT ON COLUMN src.src_opr_config_item_f.group_code IS '页面分组编码（取配置键第一段，如 agent / compression）';
COMMENT ON COLUMN src.src_opr_config_item_f.group_name IS '页面分组中文名';
COMMENT ON COLUMN src.src_opr_config_item_f.config_name IS '配置项中文名称';
COMMENT ON COLUMN src.src_opr_config_item_f.config_desc IS '配置项中文说明';
COMMENT ON COLUMN src.src_opr_config_item_f.value_type IS '取值类型：bool-布尔 int-整数 float-小数 str-字符串 enum-枚举 list-列表 map-键值对 secret-密钥引用';
COMMENT ON COLUMN src.src_opr_config_item_f.default_value IS '默认值（JSON 编码）';
COMMENT ON COLUMN src.src_opr_config_item_f.value_range IS '取值范围：{min,max} 或 {enum:[{value,label}]} 或 {pattern}';
COMMENT ON COLUMN src.src_opr_config_item_f.reload_type IS '生效方式：0-立即热加载 1-下一会话生效（保护提示缓存） 2-需重启服务';
COMMENT ON COLUMN src.src_opr_config_item_f.scope_type IS '可配置层级：0-仅平台 1-平台与租户均可';
COMMENT ON COLUMN src.src_opr_config_item_f.sort_qty IS '组内排序号';
COMMENT ON COLUMN src.src_opr_config_item_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_config_item_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_config_item_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_config_item_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_config_item_f.update_time IS '最后更新时间';
ALTER TABLE src.src_opr_config_item_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_config_item_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_config_item_f;
CREATE POLICY p_tenant_read ON src.src_opr_config_item_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_config_item_f;
CREATE POLICY p_tenant_write ON src.src_opr_config_item_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_opr_config_value_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  config_key varchar(256) NOT NULL,
  config_value jsonb,
  value_version_qty int NOT NULL DEFAULT 1,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, config_key)
);
COMMENT ON TABLE src.src_opr_config_value_f IS '租户配置覆盖值（平台级值以 config.yaml 为准，本表只存租户覆盖）';
COMMENT ON COLUMN src.src_opr_config_value_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_config_value_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_config_value_f.config_key IS '配置键，关联 src.src_opr_config_item_f.config_key';
COMMENT ON COLUMN src.src_opr_config_value_f.config_value IS '覆盖值（JSON 编码）';
COMMENT ON COLUMN src.src_opr_config_value_f.value_version_qty IS '版本号，每次保存 +1，用于乐观锁';
COMMENT ON COLUMN src.src_opr_config_value_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_config_value_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_config_value_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_config_value_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_config_value_f.update_time IS '最后更新时间';
ALTER TABLE src.src_opr_config_value_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_config_value_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_config_value_f;
CREATE POLICY p_tenant_read ON src.src_opr_config_value_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_config_value_f;
CREATE POLICY p_tenant_write ON src.src_opr_config_value_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_opr_config_change_log_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  config_key varchar(256) NOT NULL,
  change_scope_type smallint NOT NULL,
  before_value jsonb,
  after_value jsonb,
  apply_status smallint NOT NULL,
  config_file_sha256_code char(64),
  change_time timestamp(3) NOT NULL,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS src.src_opr_config_change_log_di_pdefault PARTITION OF src.src_opr_config_change_log_di DEFAULT;
COMMENT ON TABLE src.src_opr_config_change_log_di IS '配置变更记录（平台级与租户级修改均记录，天增量）';
COMMENT ON COLUMN src.src_opr_config_change_log_di.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_config_change_log_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_config_change_log_di.config_key IS '配置键';
COMMENT ON COLUMN src.src_opr_config_change_log_di.change_scope_type IS '变更层级：0-平台（写入 config.yaml） 1-租户（写入覆盖表）';
COMMENT ON COLUMN src.src_opr_config_change_log_di.before_value IS '变更前值（密钥类只记录是否设置）';
COMMENT ON COLUMN src.src_opr_config_change_log_di.after_value IS '变更后值（密钥类只记录是否设置）';
COMMENT ON COLUMN src.src_opr_config_change_log_di.apply_status IS '生效状态：0-已生效 1-待下一会话 2-待重启 3-生效失败';
COMMENT ON COLUMN src.src_opr_config_change_log_di.config_file_sha256_code IS '写入后 config.yaml 的 SHA-256，便于核对页面与文件一致';
COMMENT ON COLUMN src.src_opr_config_change_log_di.change_time IS '变更时间';
COMMENT ON COLUMN src.src_opr_config_change_log_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN src.src_opr_config_change_log_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_config_change_log_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_config_change_log_di.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_config_change_log_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_config_change_log_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_src_opr_config_change_log_di_0 ON src.src_opr_config_change_log_di (tenant_code, config_key, change_time);
ALTER TABLE src.src_opr_config_change_log_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_config_change_log_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_config_change_log_di;
CREATE POLICY p_tenant_read ON src.src_opr_config_change_log_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_config_change_log_di;
CREATE POLICY p_tenant_write ON src.src_opr_config_change_log_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS src.src_opr_content_batch_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  batch_no varchar(64) NOT NULL,
  batch_name varchar(128) NOT NULL,
  file_qty int NOT NULL DEFAULT 0,
  batch_status smallint NOT NULL,
  default_category_code varchar(64),
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, batch_no)
);
COMMENT ON TABLE src.src_opr_content_batch_f IS '内容上传批次表';
COMMENT ON COLUMN src.src_opr_content_batch_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_content_batch_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_content_batch_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_content_batch_f.batch_no IS '批次编号';
COMMENT ON COLUMN src.src_opr_content_batch_f.batch_name IS '批次名称';
COMMENT ON COLUMN src.src_opr_content_batch_f.file_qty IS '批次文件数';
COMMENT ON COLUMN src.src_opr_content_batch_f.batch_status IS '批次状态：0-上传中 1-已提交 2-处理中 3-全部完成 4-部分失败';
COMMENT ON COLUMN src.src_opr_content_batch_f.default_category_code IS '批次默认分类编码';
COMMENT ON COLUMN src.src_opr_content_batch_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_content_batch_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_content_batch_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_content_batch_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_content_batch_f.update_time IS '最后更新时间';
ALTER TABLE src.src_opr_content_batch_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_content_batch_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_content_batch_f;
CREATE POLICY p_tenant_read ON src.src_opr_content_batch_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_content_batch_f;
CREATE POLICY p_tenant_write ON src.src_opr_content_batch_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_content_asset_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  batch_no varchar(64),
  asset_name varchar(256) NOT NULL,
  modality_type varchar(16) NOT NULL,
  file_ext_code varchar(16) NOT NULL,
  file_size_qty bigint NOT NULL,
  file_sha256_code char(64) NOT NULL,
  storage_uri_name varchar(512) NOT NULL,
  page_qty int,
  duration_ms_qty bigint,
  stage_code smallint NOT NULL,
  stage_status smallint NOT NULL,
  category_code varchar(64),
  assignee_user_code varchar(64),
  parse_backend_name varchar(64),
  business_date date,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, asset_no)
);
COMMENT ON TABLE src.src_opr_content_asset_f IS '内容资产主表（一个上传文件一行，记录当前阶段与状态）';
COMMENT ON COLUMN src.src_opr_content_asset_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_content_asset_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_content_asset_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_content_asset_f.asset_no IS '资产编号';
COMMENT ON COLUMN src.src_opr_content_asset_f.batch_no IS '所属批次编号';
COMMENT ON COLUMN src.src_opr_content_asset_f.asset_name IS '资产名称（默认取原文件名）';
COMMENT ON COLUMN src.src_opr_content_asset_f.modality_type IS '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频';
COMMENT ON COLUMN src.src_opr_content_asset_f.file_ext_code IS '文件扩展名，如 pdf / docx / mp4';
COMMENT ON COLUMN src.src_opr_content_asset_f.file_size_qty IS '文件大小（字节）';
COMMENT ON COLUMN src.src_opr_content_asset_f.file_sha256_code IS '文件 SHA-256（租户内去重）';
COMMENT ON COLUMN src.src_opr_content_asset_f.storage_uri_name IS '原件存储地址，如 file:///data/... 或 s3://bucket/key';
COMMENT ON COLUMN src.src_opr_content_asset_f.page_qty IS '页数/工作表数（文档类）';
COMMENT ON COLUMN src.src_opr_content_asset_f.duration_ms_qty IS '时长（毫秒，音视频类）';
COMMENT ON COLUMN src.src_opr_content_asset_f.stage_code IS '当前流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用';
COMMENT ON COLUMN src.src_opr_content_asset_f.stage_status IS '当前阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消';
COMMENT ON COLUMN src.src_opr_content_asset_f.category_code IS '确认后的分类编码';
COMMENT ON COLUMN src.src_opr_content_asset_f.assignee_user_code IS '当前人工阶段处理人用户编码';
COMMENT ON COLUMN src.src_opr_content_asset_f.parse_backend_name IS '解析所用后端，如 mineru:standard / markdown / media';
COMMENT ON COLUMN src.src_opr_content_asset_f.business_date IS '业务日期（如发票日期、合同签署日期），可由抽取回填';
COMMENT ON COLUMN src.src_opr_content_asset_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_content_asset_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_content_asset_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_content_asset_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_content_asset_f.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_src_opr_content_asset_f_0 ON src.src_opr_content_asset_f (tenant_code, region_code, stage_code, stage_status);
CREATE INDEX IF NOT EXISTS idx_src_opr_content_asset_f_1 ON src.src_opr_content_asset_f (tenant_code, file_sha256_code);
ALTER TABLE src.src_opr_content_asset_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_content_asset_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_content_asset_f;
CREATE POLICY p_tenant_read ON src.src_opr_content_asset_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_content_asset_f;
CREATE POLICY p_tenant_write ON src.src_opr_content_asset_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_content_object_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  object_type varchar(32) NOT NULL,
  storage_uri_name varchar(512) NOT NULL,
  mime_type varchar(128),
  file_size_qty bigint,
  page_no int,
  time_ms_qty bigint,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);
COMMENT ON TABLE src.src_opr_content_object_f IS '内容派生文件表（转换 PDF、页面图、裁切图、关键帧、音轨、字幕、导出包等）';
COMMENT ON COLUMN src.src_opr_content_object_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_content_object_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_content_object_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_content_object_f.asset_no IS '所属资产编号';
COMMENT ON COLUMN src.src_opr_content_object_f.object_type IS '派生类型：page_image-页面图 crop_image-版面裁切图 frame-视频关键帧 audio_track-音轨 subtitle_srt-SRT字幕 subtitle_vtt-VTT字幕 mineru_raw-MinerU原始产物包 export-导出包';
COMMENT ON COLUMN src.src_opr_content_object_f.storage_uri_name IS '存储地址';
COMMENT ON COLUMN src.src_opr_content_object_f.mime_type IS 'MIME 类型，开放取值，遵循 IANA 媒体类型，如 application/pdf';
COMMENT ON COLUMN src.src_opr_content_object_f.file_size_qty IS '文件大小（字节）';
COMMENT ON COLUMN src.src_opr_content_object_f.page_no IS '关联页码（从 1 开始）';
COMMENT ON COLUMN src.src_opr_content_object_f.time_ms_qty IS '关联时间点（毫秒，关键帧用）';
COMMENT ON COLUMN src.src_opr_content_object_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_content_object_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_content_object_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_content_object_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_content_object_f.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_src_opr_content_object_f_0 ON src.src_opr_content_object_f (tenant_code, asset_no, object_type);
ALTER TABLE src.src_opr_content_object_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_content_object_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_content_object_f;
CREATE POLICY p_tenant_read ON src.src_opr_content_object_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_content_object_f;
CREATE POLICY p_tenant_write ON src.src_opr_content_object_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_content_media_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  duration_ms_qty bigint NOT NULL,
  container_name varchar(32),
  video_codec_name varchar(32),
  audio_codec_name varchar(32),
  width_qty int,
  height_qty int,
  frame_per_sec_qty decimal(8,3),
  sample_rate_qty int,
  channel_qty smallint,
  speech_lang_code varchar(16),
  frame_qty int,
  segment_qty int,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, asset_no)
);
COMMENT ON TABLE src.src_opr_content_media_f IS '音视频元数据表（ffprobe 抽取）';
COMMENT ON COLUMN src.src_opr_content_media_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_content_media_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_content_media_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_content_media_f.asset_no IS '资产编号';
COMMENT ON COLUMN src.src_opr_content_media_f.duration_ms_qty IS '时长（毫秒）';
COMMENT ON COLUMN src.src_opr_content_media_f.container_name IS '封装格式，如 mp4 / mkv / mp3 / wav';
COMMENT ON COLUMN src.src_opr_content_media_f.video_codec_name IS '视频编码，如 h264';
COMMENT ON COLUMN src.src_opr_content_media_f.audio_codec_name IS '音频编码，如 aac';
COMMENT ON COLUMN src.src_opr_content_media_f.width_qty IS '画面宽度（像素）';
COMMENT ON COLUMN src.src_opr_content_media_f.height_qty IS '画面高度（像素）';
COMMENT ON COLUMN src.src_opr_content_media_f.frame_per_sec_qty IS '帧率（帧/秒）';
COMMENT ON COLUMN src.src_opr_content_media_f.sample_rate_qty IS '音频采样率（Hz）';
COMMENT ON COLUMN src.src_opr_content_media_f.channel_qty IS '音频声道数';
COMMENT ON COLUMN src.src_opr_content_media_f.speech_lang_code IS '识别出的语种，如 zh / en';
COMMENT ON COLUMN src.src_opr_content_media_f.frame_qty IS '抽取关键帧数';
COMMENT ON COLUMN src.src_opr_content_media_f.segment_qty IS '转写语音片段数';
COMMENT ON COLUMN src.src_opr_content_media_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_content_media_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_content_media_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_content_media_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_content_media_f.update_time IS '最后更新时间';
ALTER TABLE src.src_opr_content_media_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_content_media_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_content_media_f;
CREATE POLICY p_tenant_read ON src.src_opr_content_media_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_content_media_f;
CREATE POLICY p_tenant_write ON src.src_opr_content_media_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_content_stage_task_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  task_no varchar(64) NOT NULL,
  asset_no varchar(64) NOT NULL,
  stage_code smallint NOT NULL,
  task_status smallint NOT NULL,
  attempt_qty int NOT NULL DEFAULT 0,
  max_attempt_qty int NOT NULL DEFAULT 3,
  next_retry_time timestamp(3),
  queue_message_code varchar(64),
  worker_name varchar(128),
  operator_user_code varchar(64),
  start_time timestamp(3),
  end_time timestamp(3),
  error_code varchar(64),
  error_msg text,
  task_param_json jsonb,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, task_no)
);
COMMENT ON TABLE src.src_opr_content_stage_task_f IS '流水线阶段任务表（每个资产每个阶段一次执行一行，含重试）';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.task_no IS '任务编号';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.asset_no IS '资产编号';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.stage_code IS '流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.task_status IS '任务阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.attempt_qty IS '已执行次数';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.max_attempt_qty IS '最大执行次数（来自 config.yaml pipeline.retry）';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.next_retry_time IS '下次重试时间（指数退避）';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.queue_message_code IS 'Redis Stream 消息 ID';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.worker_name IS '执行该任务的 worker 名称';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.operator_user_code IS '人工阶段的操作人用户编码；自动阶段为 system';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.start_time IS '开始时间';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.end_time IS '结束时间';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.error_code IS '失败错误码，如 MINERU_TIMEOUT / ASR_FAILED';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.error_msg IS '失败详情';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.task_param_json IS '任务参数（解析档位、OCR 语言等）';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_content_stage_task_f.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_src_opr_content_stage_task_f_0 ON src.src_opr_content_stage_task_f (tenant_code, asset_no, stage_code);
CREATE INDEX IF NOT EXISTS idx_src_opr_content_stage_task_f_1 ON src.src_opr_content_stage_task_f (task_status, next_retry_time);
ALTER TABLE src.src_opr_content_stage_task_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_content_stage_task_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_content_stage_task_f;
CREATE POLICY p_tenant_read ON src.src_opr_content_stage_task_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_content_stage_task_f;
CREATE POLICY p_tenant_write ON src.src_opr_content_stage_task_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_content_stage_log_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  task_no varchar(64),
  from_stage_code smallint NOT NULL,
  from_stage_status smallint NOT NULL,
  to_stage_code smallint NOT NULL,
  to_stage_status smallint NOT NULL,
  operator_user_code varchar(64) NOT NULL,
  change_reason varchar(512),
  change_time timestamp(3) NOT NULL,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS src.src_opr_content_stage_log_di_pdefault PARTITION OF src.src_opr_content_stage_log_di DEFAULT;
COMMENT ON TABLE src.src_opr_content_stage_log_di IS '资产状态流转日志（每次状态变化一行，天增量）';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.asset_no IS '资产编号';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.task_no IS '关联任务编号';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.from_stage_code IS '变更前流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用，0-新建';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.from_stage_status IS '变更前阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.to_stage_code IS '变更后流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.to_stage_status IS '变更后阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.operator_user_code IS '操作人用户编码，系统自动为 system';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.change_reason IS '变更原因（驳回意见、失败原因、跳过说明）';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.change_time IS '变更时间';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_content_stage_log_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_src_opr_content_stage_log_di_0 ON src.src_opr_content_stage_log_di (tenant_code, asset_no, change_time);
ALTER TABLE src.src_opr_content_stage_log_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_content_stage_log_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_content_stage_log_di;
CREATE POLICY p_tenant_read ON src.src_opr_content_stage_log_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_content_stage_log_di;
CREATE POLICY p_tenant_write ON src.src_opr_content_stage_log_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_content_block_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  block_no varchar(64) NOT NULL,
  asset_no varchar(64) NOT NULL,
  block_type varchar(16) NOT NULL,
  block_order_qty int NOT NULL,
  parent_block_no varchar(64),
  page_no int,
  bbox_json jsonb,
  title_level_qty smallint,
  start_ms_qty bigint,
  end_ms_qty bigint,
  speaker_code varchar(32),
  origin_text text,
  review_text text,
  table_html text,
  table_json jsonb,
  image_object_id bigint,
  confidence_rate decimal(5,4),
  review_status smallint NOT NULL DEFAULT 0,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, block_no)
);
COMMENT ON TABLE src.src_opr_content_block_f IS '内容块工作表（解析结果的结构化内容，供人工核对、标注；入库后同步到 DWD）';
COMMENT ON COLUMN src.src_opr_content_block_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_content_block_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_content_block_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_content_block_f.block_no IS '内容块编号';
COMMENT ON COLUMN src.src_opr_content_block_f.asset_no IS '资产编号';
COMMENT ON COLUMN src.src_opr_content_block_f.block_type IS '内容块类型：title-标题 text-正文 list-列表 table-表格 image-图片 equation-公式 code-代码 header-页眉 footer-页脚 sheet-工作表 frame-视频关键帧 speech-语音片段';
COMMENT ON COLUMN src.src_opr_content_block_f.block_order_qty IS '阅读顺序号（资产内从 1 开始）';
COMMENT ON COLUMN src.src_opr_content_block_f.parent_block_no IS '父块编号（表格标题/图片说明挂在表格/图片上）';
COMMENT ON COLUMN src.src_opr_content_block_f.page_no IS '页码/工作表序号（从 1 开始）';
COMMENT ON COLUMN src.src_opr_content_block_f.bbox_json IS '版面坐标 [x0,y0,x1,y1]（页面归一化到 0~1000）';
COMMENT ON COLUMN src.src_opr_content_block_f.title_level_qty IS '标题层级（block_type=title 时 1~6）';
COMMENT ON COLUMN src.src_opr_content_block_f.start_ms_qty IS '开始时间（毫秒，speech/frame 用）';
COMMENT ON COLUMN src.src_opr_content_block_f.end_ms_qty IS '结束时间（毫秒，speech 用）';
COMMENT ON COLUMN src.src_opr_content_block_f.speaker_code IS '说话人标识（语音片段，若开启说话人分离）';
COMMENT ON COLUMN src.src_opr_content_block_f.origin_text IS '解析原始文本（不可改，便于对比）';
COMMENT ON COLUMN src.src_opr_content_block_f.review_text IS '人工核对后文本，为空表示与原文一致';
COMMENT ON COLUMN src.src_opr_content_block_f.table_html IS '表格 HTML（block_type=table/sheet）';
COMMENT ON COLUMN src.src_opr_content_block_f.table_json IS '表格结构化数据（二维数组，含合并单元格信息）';
COMMENT ON COLUMN src.src_opr_content_block_f.image_object_id IS '关联图片派生文件 id（src.src_opr_content_object_f.id）';
COMMENT ON COLUMN src.src_opr_content_block_f.confidence_rate IS '解析/识别置信度 0~1';
COMMENT ON COLUMN src.src_opr_content_block_f.review_status IS '核对状态：0-未核对 1-已确认无误 2-已修改 3-标记删除';
COMMENT ON COLUMN src.src_opr_content_block_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_content_block_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_content_block_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_content_block_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_content_block_f.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_src_opr_content_block_f_0 ON src.src_opr_content_block_f (tenant_code, asset_no, block_order_qty);
ALTER TABLE src.src_opr_content_block_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_content_block_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_content_block_f;
CREATE POLICY p_tenant_read ON src.src_opr_content_block_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_content_block_f;
CREATE POLICY p_tenant_write ON src.src_opr_content_block_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_media_align_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  frame_block_no varchar(64) NOT NULL,
  speech_block_no varchar(64) NOT NULL,
  overlap_ms_qty bigint NOT NULL,
  align_type smallint NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, frame_block_no, speech_block_no)
);
COMMENT ON TABLE src.src_opr_media_align_f IS '音视频对齐表（关键帧与语音片段按时间重叠对齐，用于图文字幕联合检索）';
COMMENT ON COLUMN src.src_opr_media_align_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_media_align_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_media_align_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_media_align_f.asset_no IS '资产编号';
COMMENT ON COLUMN src.src_opr_media_align_f.frame_block_no IS '关键帧内容块编号（block_type=frame）';
COMMENT ON COLUMN src.src_opr_media_align_f.speech_block_no IS '语音片段内容块编号（block_type=speech）';
COMMENT ON COLUMN src.src_opr_media_align_f.overlap_ms_qty IS '两者时间重叠时长（毫秒）';
COMMENT ON COLUMN src.src_opr_media_align_f.align_type IS '对齐方式：1-时间重叠 2-最近邻（帧落在片段间隙时） 3-人工调整';
COMMENT ON COLUMN src.src_opr_media_align_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_media_align_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_media_align_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_media_align_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_media_align_f.update_time IS '最后更新时间';
ALTER TABLE src.src_opr_media_align_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_media_align_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_media_align_f;
CREATE POLICY p_tenant_read ON src.src_opr_media_align_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_media_align_f;
CREATE POLICY p_tenant_write ON src.src_opr_media_align_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_content_label_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  label_no varchar(64) NOT NULL,
  asset_no varchar(64) NOT NULL,
  block_no varchar(64),
  label_code varchar(64) NOT NULL,
  label_type smallint NOT NULL,
  label_value text,
  char_start_qty int,
  char_end_qty int,
  bbox_json jsonb,
  start_ms_qty bigint,
  end_ms_qty bigint,
  cell_json jsonb,
  label_source_type smallint NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, label_no)
);
COMMENT ON TABLE src.src_opr_content_label_f IS '标注记录表';
COMMENT ON COLUMN src.src_opr_content_label_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_content_label_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_content_label_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_content_label_f.label_no IS '标注编号';
COMMENT ON COLUMN src.src_opr_content_label_f.asset_no IS '资产编号';
COMMENT ON COLUMN src.src_opr_content_label_f.block_no IS '内容块编号，整篇标签为空';
COMMENT ON COLUMN src.src_opr_content_label_f.label_code IS '标签编码（dim.dim_opr_label_schema_f）';
COMMENT ON COLUMN src.src_opr_content_label_f.label_type IS '标注形态：1-文本实体 2-图像区域框 3-音视频时间片 4-整篇标签 5-表格单元格';
COMMENT ON COLUMN src.src_opr_content_label_f.label_value IS '标注值（实体文本或用户填写值）';
COMMENT ON COLUMN src.src_opr_content_label_f.char_start_qty IS '文本实体起始字符偏移';
COMMENT ON COLUMN src.src_opr_content_label_f.char_end_qty IS '文本实体结束字符偏移（不含）';
COMMENT ON COLUMN src.src_opr_content_label_f.bbox_json IS '图像区域框 [x0,y0,x1,y1]';
COMMENT ON COLUMN src.src_opr_content_label_f.start_ms_qty IS '时间片开始（毫秒）';
COMMENT ON COLUMN src.src_opr_content_label_f.end_ms_qty IS '时间片结束（毫秒）';
COMMENT ON COLUMN src.src_opr_content_label_f.cell_json IS '表格单元格坐标 {row,col}';
COMMENT ON COLUMN src.src_opr_content_label_f.label_source_type IS '标注来源：0-人工 1-模型预标注 2-模型预标注经人工确认';
COMMENT ON COLUMN src.src_opr_content_label_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_content_label_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_content_label_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_content_label_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_content_label_f.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_src_opr_content_label_f_0 ON src.src_opr_content_label_f (tenant_code, asset_no);
ALTER TABLE src.src_opr_content_label_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_content_label_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_content_label_f;
CREATE POLICY p_tenant_read ON src.src_opr_content_label_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_content_label_f;
CREATE POLICY p_tenant_write ON src.src_opr_content_label_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_content_category_rel_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  category_code varchar(64) NOT NULL,
  classify_source_type smallint NOT NULL,
  confidence_rate decimal(5,4),
  confirm_status smallint NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, asset_no, category_code, classify_source_type)
);
COMMENT ON TABLE src.src_opr_content_category_rel_f IS '资产分类关系表（含模型建议与人工确认）';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.asset_no IS '资产编号';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.category_code IS '分类编码';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.classify_source_type IS '分类来源：0-人工 1-规则 2-模型建议';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.confidence_rate IS '模型建议置信度 0~1';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.confirm_status IS '确认状态：0-待确认 1-已采纳 2-已否决';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_content_category_rel_f.update_time IS '最后更新时间';
ALTER TABLE src.src_opr_content_category_rel_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_content_category_rel_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_content_category_rel_f;
CREATE POLICY p_tenant_read ON src.src_opr_content_category_rel_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_content_category_rel_f;
CREATE POLICY p_tenant_write ON src.src_opr_content_category_rel_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_knowledge_base_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  kb_code varchar(64) NOT NULL,
  kb_name varchar(128) NOT NULL,
  kb_desc varchar(512),
  embedding_model_name varchar(128) NOT NULL,
  embedding_dim_qty int NOT NULL,
  kb_status smallint NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, kb_code)
);
COMMENT ON TABLE src.src_opr_knowledge_base_f IS '知识库表（入库内容的应用载体，挂载到租户智能体）';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.kb_code IS '知识库编码';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.kb_name IS '知识库名称';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.kb_desc IS '知识库说明（提供给智能体的工具描述）';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.embedding_model_name IS '向量模型名称';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.embedding_dim_qty IS '向量维度';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.kb_status IS '知识库状态：0-启用 1-停用';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_knowledge_base_f.update_time IS '最后更新时间';
ALTER TABLE src.src_opr_knowledge_base_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_knowledge_base_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_knowledge_base_f;
CREATE POLICY p_tenant_read ON src.src_opr_knowledge_base_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_knowledge_base_f;
CREATE POLICY p_tenant_write ON src.src_opr_knowledge_base_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_content_application_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  apply_type smallint NOT NULL,
  target_code varchar(256) NOT NULL,
  apply_status smallint NOT NULL,
  apply_result_json jsonb,
  apply_time timestamp(3),
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, asset_no, apply_type, target_code)
);
COMMENT ON TABLE src.src_opr_content_application_f IS '内容应用记录表（入库内容被投放到哪些应用）';
COMMENT ON COLUMN src.src_opr_content_application_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_content_application_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_content_application_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_content_application_f.asset_no IS '资产编号';
COMMENT ON COLUMN src.src_opr_content_application_f.apply_type IS '应用类型：1-知识库检索 2-业务表回填 3-导出下载 4-智能体会话附件';
COMMENT ON COLUMN src.src_opr_content_application_f.target_code IS '应用目标：知识库编码 / 回填表（库名.表名） / 导出格式';
COMMENT ON COLUMN src.src_opr_content_application_f.apply_status IS '应用状态：0-待执行 1-执行中 2-成功 3-失败 4-已撤回';
COMMENT ON COLUMN src.src_opr_content_application_f.apply_result_json IS '执行结果（切片数、回填行数、导出地址等）';
COMMENT ON COLUMN src.src_opr_content_application_f.apply_time IS '执行完成时间';
COMMENT ON COLUMN src.src_opr_content_application_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_content_application_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_content_application_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_content_application_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_content_application_f.update_time IS '最后更新时间';
ALTER TABLE src.src_opr_content_application_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_content_application_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_content_application_f;
CREATE POLICY p_tenant_read ON src.src_opr_content_application_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_content_application_f;
CREATE POLICY p_tenant_write ON src.src_opr_content_application_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_portal_suggest_question_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  question_text varchar(500) NOT NULL,
  category_name varchar(64),
  kb_code varchar(64),
  sort_qty int NOT NULL DEFAULT 0,
  show_status smallint NOT NULL DEFAULT 0,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);
COMMENT ON TABLE src.src_opr_portal_suggest_question_f IS '用户端推荐问题（管理端维护，展示在问答首页）';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.question_text IS '推荐问题文本';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.category_name IS '问题下方显示的资料类别标签，如 产品资料 / 服务规范';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.kb_code IS '点击后限定检索的知识库编码，为空表示全部可用资料';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.sort_qty IS '排序号，越小越靠前';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.show_status IS '展示状态：0-展示 1-隐藏';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_portal_suggest_question_f.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_src_opr_portal_suggest_question_f_0 ON src.src_opr_portal_suggest_question_f (tenant_code, show_status, sort_qty);
ALTER TABLE src.src_opr_portal_suggest_question_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_portal_suggest_question_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_portal_suggest_question_f;
CREATE POLICY p_tenant_read ON src.src_opr_portal_suggest_question_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_portal_suggest_question_f;
CREATE POLICY p_tenant_write ON src.src_opr_portal_suggest_question_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS src.src_opr_portal_feedback_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  session_code varchar(128) NOT NULL,
  message_code varchar(128) NOT NULL,
  user_code varchar(64) NOT NULL,
  feedback_type smallint NOT NULL,
  feedback_text varchar(1000),
  question_text text,
  cite_asset_no_list varchar(1024),
  handle_status smallint NOT NULL DEFAULT 0,
  feedback_time timestamp(3) NOT NULL,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS src.src_opr_portal_feedback_di_pdefault PARTITION OF src.src_opr_portal_feedback_di DEFAULT;
COMMENT ON TABLE src.src_opr_portal_feedback_di IS '用户端问答反馈（有帮助/没帮助，供管理端问答会话审计，天增量）';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.id IS '自增主键';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.session_code IS 'hermes 会话 ID';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.message_code IS '被评价的回答消息 ID';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.user_code IS '反馈用户编码（用户端问答用户）';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.feedback_type IS '反馈类型：1-有帮助 2-没帮助';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.feedback_text IS '补充说明';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.question_text IS '对应的问题原文（便于审计检索）';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.cite_asset_no_list IS '回答引用的资产编号，逗号分隔';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.handle_status IS '处理状态：0-待处理 1-已处理 2-已忽略';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.feedback_time IS '反馈时间';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.create_time IS '创建时间';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN src.src_opr_portal_feedback_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_src_opr_portal_feedback_di_0 ON src.src_opr_portal_feedback_di (tenant_code, handle_status, feedback_time);
ALTER TABLE src.src_opr_portal_feedback_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE src.src_opr_portal_feedback_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON src.src_opr_portal_feedback_di;
CREATE POLICY p_tenant_read ON src.src_opr_portal_feedback_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON src.src_opr_portal_feedback_di;
CREATE POLICY p_tenant_write ON src.src_opr_portal_feedback_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

-- ======================================================================
-- ods
-- ======================================================================

CREATE TABLE IF NOT EXISTS ods.ods_opr_mineru_parse_result_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  task_no varchar(64) NOT NULL,
  parser_type varchar(32) NOT NULL,
  mineru_version_name varchar(32),
  mineru_tier_name varchar(16),
  mineru_task_code varchar(64),
  middle_json_uri_name varchar(512),
  markdown_uri_name varchar(512),
  structured_json_uri_name varchar(512),
  block_qty int,
  parse_duration_ms_qty bigint,
  parse_status smallint NOT NULL,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS ods.ods_opr_mineru_parse_result_di_pdefault PARTITION OF ods.ods_opr_mineru_parse_result_di DEFAULT;
COMMENT ON TABLE ods.ods_opr_mineru_parse_result_di IS 'MinerU 原始解析结果（每次解析一行，分区日期=解析日期）';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.id IS '自增主键';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.asset_no IS '资产编号';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.task_no IS '解析任务编号';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.parser_type IS '解析通道：mineru-MinerU文档解析 markdown-Markdown直读 media-音视频管线';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.mineru_version_name IS 'MinerU 版本号';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.mineru_tier_name IS 'MinerU 档位：flash / basic / standard / advanced';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.mineru_task_code IS 'MinerU 服务侧任务 ID';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.middle_json_uri_name IS 'middle_json.json 存储地址';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.markdown_uri_name IS 'markdown.md 存储地址';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.structured_json_uri_name IS 'structured_content.json 存储地址';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.block_qty IS '解析出的内容块数';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.parse_duration_ms_qty IS '解析耗时（毫秒）';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.parse_status IS '解析结果：0-成功 1-失败 2-部分成功';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.create_time IS '创建时间';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN ods.ods_opr_mineru_parse_result_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_ods_opr_mineru_parse_result_di_0 ON ods.ods_opr_mineru_parse_result_di (tenant_code, asset_no);
ALTER TABLE ods.ods_opr_mineru_parse_result_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE ods.ods_opr_mineru_parse_result_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON ods.ods_opr_mineru_parse_result_di;
CREATE POLICY p_tenant_read ON ods.ods_opr_mineru_parse_result_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON ods.ods_opr_mineru_parse_result_di;
CREATE POLICY p_tenant_write ON ods.ods_opr_mineru_parse_result_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS ods.ods_opr_media_transcript_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  task_no varchar(64) NOT NULL,
  segment_order_qty int NOT NULL,
  start_ms_qty bigint NOT NULL,
  end_ms_qty bigint NOT NULL,
  speaker_code varchar(32),
  segment_text text NOT NULL,
  word_json jsonb,
  avg_logprob_score decimal(8,4),
  asr_model_name varchar(64),
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS ods.ods_opr_media_transcript_di_pdefault PARTITION OF ods.ods_opr_media_transcript_di DEFAULT;
COMMENT ON TABLE ods.ods_opr_media_transcript_di IS '语音转写原始片段（ASR 原样输出，分区日期=转写日期）';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.id IS '自增主键';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.asset_no IS '资产编号';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.task_no IS '转写任务编号';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.segment_order_qty IS '片段顺序号';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.start_ms_qty IS '开始时间（毫秒）';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.end_ms_qty IS '结束时间（毫秒）';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.speaker_code IS '说话人标识';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.segment_text IS '转写文本';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.word_json IS '词级时间戳 [{word,start_ms,end_ms,prob}]';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.avg_logprob_score IS '平均对数概率（ASR 置信参考，越接近 0 越可信）';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.asr_model_name IS 'ASR 模型名称，如 faster-whisper-large-v3';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.create_time IS '创建时间';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN ods.ods_opr_media_transcript_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_ods_opr_media_transcript_di_0 ON ods.ods_opr_media_transcript_di (tenant_code, asset_no);
ALTER TABLE ods.ods_opr_media_transcript_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE ods.ods_opr_media_transcript_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON ods.ods_opr_media_transcript_di;
CREATE POLICY p_tenant_read ON ods.ods_opr_media_transcript_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON ods.ods_opr_media_transcript_di;
CREATE POLICY p_tenant_write ON ods.ods_opr_media_transcript_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS ods.ods_opr_media_frame_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  task_no varchar(64) NOT NULL,
  frame_order_qty int NOT NULL,
  time_ms_qty bigint NOT NULL,
  extract_type smallint NOT NULL,
  scene_change_score decimal(6,4),
  storage_uri_name varchar(512) NOT NULL,
  ocr_text text,
  caption_text text,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS ods.ods_opr_media_frame_di_pdefault PARTITION OF ods.ods_opr_media_frame_di DEFAULT;
COMMENT ON TABLE ods.ods_opr_media_frame_di IS '视频抽帧原始记录（分区日期=抽帧日期）';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.id IS '自增主键';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.asset_no IS '资产编号';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.task_no IS '抽帧任务编号';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.frame_order_qty IS '帧顺序号';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.time_ms_qty IS '帧时间点（毫秒）';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.extract_type IS '抽帧方式：1-场景切换 2-固定间隔 3-人工补帧';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.scene_change_score IS '场景切换分值 0~1';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.storage_uri_name IS '帧图片存储地址';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.ocr_text IS '帧图片经 MinerU 识别的文字';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.caption_text IS '视觉模型生成的画面描述（可选）';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.create_time IS '创建时间';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN ods.ods_opr_media_frame_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_ods_opr_media_frame_di_0 ON ods.ods_opr_media_frame_di (tenant_code, asset_no);
ALTER TABLE ods.ods_opr_media_frame_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE ods.ods_opr_media_frame_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON ods.ods_opr_media_frame_di;
CREATE POLICY p_tenant_read ON ods.ods_opr_media_frame_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON ods.ods_opr_media_frame_di;
CREATE POLICY p_tenant_write ON ods.ods_opr_media_frame_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS ods.ods_opr_agent_session_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  session_code varchar(128) NOT NULL,
  user_code varchar(64) NOT NULL,
  model_name varchar(128),
  provider_name varchar(64),
  input_token_qty bigint NOT NULL DEFAULT 0,
  output_token_qty bigint NOT NULL DEFAULT 0,
  cache_read_token_qty bigint NOT NULL DEFAULT 0,
  cost_amt decimal(18,6),
  session_start_time timestamp(3),
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date),
  UNIQUE (tenant_code, session_code, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS ods.ods_opr_agent_session_di_pdefault PARTITION OF ods.ods_opr_agent_session_di DEFAULT;
COMMENT ON TABLE ods.ods_opr_agent_session_di IS '智能体会话用量原始记录（从各租户 hermes profile 同步，分区日期=会话日期）';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.id IS '自增主键';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.session_code IS 'hermes 会话 ID';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.user_code IS '发起用户编码';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.model_name IS '使用的模型';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.provider_name IS '模型提供商';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.input_token_qty IS '输入 token 数';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.output_token_qty IS '输出 token 数';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.cache_read_token_qty IS '缓存命中 token 数';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.cost_amt IS '估算费用（美元）';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.session_start_time IS '会话开始时间';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.create_time IS '创建时间';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN ods.ods_opr_agent_session_di.update_time IS '最后更新时间';
ALTER TABLE ods.ods_opr_agent_session_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE ods.ods_opr_agent_session_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON ods.ods_opr_agent_session_di;
CREATE POLICY p_tenant_read ON ods.ods_opr_agent_session_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON ods.ods_opr_agent_session_di;
CREATE POLICY p_tenant_write ON ods.ods_opr_agent_session_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS ods.ods_fin_invoice_extract_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  invoice_no varchar(64),
  invoice_code varchar(64),
  invoice_type varchar(32),
  invoice_date date,
  seller_name varchar(256),
  buyer_name varchar(256),
  invoice_amt decimal(18,2),
  tax_amt decimal(18,2),
  tax_rate decimal(6,4),
  verify_status smallint NOT NULL DEFAULT 0,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS ods.ods_fin_invoice_extract_di_pdefault PARTITION OF ods.ods_fin_invoice_extract_di DEFAULT;
COMMENT ON TABLE ods.ods_fin_invoice_extract_di IS '发票抽取原始数据（发票类内容入库回填，分区日期=开票日期）';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.id IS '自增主键';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.asset_no IS '来源资产编号';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.invoice_no IS '发票号码';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.invoice_code IS '发票代码';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.invoice_type IS '发票类型：vat_special-增值税专票 vat_normal-增值税普票 e_invoice-全电发票 itinerary-航空行程单 other-其他';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.invoice_date IS '开票日期';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.seller_name IS '销售方名称';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.buyer_name IS '购买方名称';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.invoice_amt IS '价税合计金额';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.tax_amt IS '税额';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.tax_rate IS '税率';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.verify_status IS '人工核验状态：0-待核验 1-已核验 2-有误';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.create_time IS '创建时间';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN ods.ods_fin_invoice_extract_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_ods_fin_invoice_extract_di_0 ON ods.ods_fin_invoice_extract_di (tenant_code, invoice_no);
ALTER TABLE ods.ods_fin_invoice_extract_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE ods.ods_fin_invoice_extract_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON ods.ods_fin_invoice_extract_di;
CREATE POLICY p_tenant_read ON ods.ods_fin_invoice_extract_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON ods.ods_fin_invoice_extract_di;
CREATE POLICY p_tenant_write ON ods.ods_fin_invoice_extract_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS ods.ods_sup_contract_extract_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  contract_no varchar(64),
  sup_name varchar(256),
  sup_type varchar(32),
  sign_date date,
  start_date date,
  end_date date,
  settlement_type varchar(32),
  net_price_amt decimal(18,2),
  commission_rate decimal(6,4),
  verify_status smallint NOT NULL DEFAULT 0,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS ods.ods_sup_contract_extract_di_pdefault PARTITION OF ods.ods_sup_contract_extract_di DEFAULT;
COMMENT ON TABLE ods.ods_sup_contract_extract_di IS '供应商合同抽取原始数据（分区日期=签署日期）';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.id IS '自增主键';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.asset_no IS '来源资产编号';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.contract_no IS '合同编号';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.sup_name IS '供应商名称';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.sup_type IS '供应商类型：hotel-酒店 airline-航司 ground-地接 attraction-景点 other-其他';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.sign_date IS '签署日期';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.start_date IS '合作开始日期';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.end_date IS '合作结束日期';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.settlement_type IS '结算方式：prepay-预付 monthly-月结 per_order-单结';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.net_price_amt IS '结算价金额（合同主报价）';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.commission_rate IS '佣金比例';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.verify_status IS '人工核验状态：0-待核验 1-已核验 2-有误';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.create_time IS '创建时间';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN ods.ods_sup_contract_extract_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_ods_sup_contract_extract_di_0 ON ods.ods_sup_contract_extract_di (tenant_code, contract_no);
ALTER TABLE ods.ods_sup_contract_extract_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE ods.ods_sup_contract_extract_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON ods.ods_sup_contract_extract_di;
CREATE POLICY p_tenant_read ON ods.ods_sup_contract_extract_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON ods.ods_sup_contract_extract_di;
CREATE POLICY p_tenant_write ON ods.ods_sup_contract_extract_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS ods.ods_prd_product_extract_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  prd_name varchar(256),
  prd_type varchar(32),
  attraction_name varchar(256),
  tour_day_qty int,
  adult_price_amt decimal(18,2),
  child_price_amt decimal(18,2),
  valid_start_date date,
  valid_end_date date,
  verify_status smallint NOT NULL DEFAULT 0,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS ods.ods_prd_product_extract_di_pdefault PARTITION OF ods.ods_prd_product_extract_di DEFAULT;
COMMENT ON TABLE ods.ods_prd_product_extract_di IS '产品资料抽取原始数据（线路/酒店/门票宣传资料，分区日期=资料生效日期）';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.id IS '自增主键';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.asset_no IS '来源资产编号';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.prd_name IS '产品名称';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.prd_type IS '产品类型：flight-机票 hotel-酒店 tour-跟团游 combo-组合产品 attraction-景点门票';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.attraction_name IS '主要景点名称';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.tour_day_qty IS '行程天数';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.adult_price_amt IS '成人价';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.child_price_amt IS '儿童价';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.valid_start_date IS '有效期开始日期';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.valid_end_date IS '有效期结束日期';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.verify_status IS '人工核验状态：0-待核验 1-已核验 2-有误';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.create_time IS '创建时间';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN ods.ods_prd_product_extract_di.update_time IS '最后更新时间';
ALTER TABLE ods.ods_prd_product_extract_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE ods.ods_prd_product_extract_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON ods.ods_prd_product_extract_di;
CREATE POLICY p_tenant_read ON ods.ods_prd_product_extract_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON ods.ods_prd_product_extract_di;
CREATE POLICY p_tenant_write ON ods.ods_prd_product_extract_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

-- ======================================================================
-- dwd
-- ======================================================================

CREATE TABLE IF NOT EXISTS dwd.dwd_opr_content_block_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  block_no varchar(64) NOT NULL,
  asset_no varchar(64) NOT NULL,
  modality_type varchar(16) NOT NULL,
  category_code varchar(64) NOT NULL,
  block_type varchar(16) NOT NULL,
  block_order_qty int NOT NULL,
  page_no int,
  start_ms_qty bigint,
  end_ms_qty bigint,
  final_text text,
  table_json jsonb,
  label_json jsonb,
  text_edit_status smallint NOT NULL,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS dwd.dwd_opr_content_block_di_pdefault PARTITION OF dwd.dwd_opr_content_block_di DEFAULT;
COMMENT ON TABLE dwd.dwd_opr_content_block_di IS '入库内容块明细（核对后最终文本，分区日期=入库日期）';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.id IS '自增主键';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.block_no IS '内容块编号';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.asset_no IS '资产编号';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.modality_type IS '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.category_code IS '资产分类编码';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.block_type IS '内容块类型：title-标题 text-正文 list-列表 table-表格 image-图片 equation-公式 code-代码 header-页眉 footer-页脚 sheet-工作表 frame-视频关键帧 speech-语音片段';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.block_order_qty IS '阅读顺序号';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.page_no IS '页码';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.start_ms_qty IS '开始时间（毫秒）';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.end_ms_qty IS '结束时间（毫秒）';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.final_text IS '最终文本（核对文本优先，否则原始文本）';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.table_json IS '表格结构化数据';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.label_json IS '该块上的标注汇总 [{label_code,value,...}]';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.text_edit_status IS '是否经人工修改：0-否 1-是';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.create_time IS '创建时间';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dwd.dwd_opr_content_block_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_dwd_opr_content_block_di_0 ON dwd.dwd_opr_content_block_di (tenant_code, asset_no);
CREATE INDEX IF NOT EXISTS idx_dwd_opr_content_block_di_1 ON dwd.dwd_opr_content_block_di (tenant_code, category_code);
ALTER TABLE dwd.dwd_opr_content_block_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE dwd.dwd_opr_content_block_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dwd.dwd_opr_content_block_di;
CREATE POLICY p_tenant_read ON dwd.dwd_opr_content_block_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dwd.dwd_opr_content_block_di;
CREATE POLICY p_tenant_write ON dwd.dwd_opr_content_block_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS dwd.dwd_opr_knowledge_chunk_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  chunk_no varchar(64) NOT NULL,
  kb_code varchar(64) NOT NULL,
  asset_no varchar(64) NOT NULL,
  block_no_list text NOT NULL,
  chunk_text text NOT NULL,
  chunk_token_qty int NOT NULL,
  page_no int,
  start_ms_qty bigint,
  embedding_vector vector,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS dwd.dwd_opr_knowledge_chunk_di_pdefault PARTITION OF dwd.dwd_opr_knowledge_chunk_di DEFAULT;
COMMENT ON TABLE dwd.dwd_opr_knowledge_chunk_di IS '知识切片明细（含向量，供智能体检索，分区日期=入库日期）';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.id IS '自增主键';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.chunk_no IS '切片编号';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.kb_code IS '知识库编码';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.asset_no IS '来源资产编号';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.block_no_list IS '组成该切片的内容块编号，逗号分隔';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.chunk_text IS '切片文本（含标题路径前缀）';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.chunk_token_qty IS '切片 token 数';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.page_no IS '起始页码';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.start_ms_qty IS '起始时间（毫秒，音视频切片）';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.embedding_vector IS '向量（PostgreSQL 使用 pgvector；MySQL 以 JSON 文本存储，不参与检索）';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.create_time IS '创建时间';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dwd.dwd_opr_knowledge_chunk_di.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_dwd_opr_knowledge_chunk_di_0 ON dwd.dwd_opr_knowledge_chunk_di (tenant_code, kb_code);
CREATE INDEX IF NOT EXISTS idx_dwd_opr_knowledge_chunk_di_1 ON dwd.dwd_opr_knowledge_chunk_di (tenant_code, asset_no);
ALTER TABLE dwd.dwd_opr_knowledge_chunk_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE dwd.dwd_opr_knowledge_chunk_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dwd.dwd_opr_knowledge_chunk_di;
CREATE POLICY p_tenant_read ON dwd.dwd_opr_knowledge_chunk_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dwd.dwd_opr_knowledge_chunk_di;
CREATE POLICY p_tenant_write ON dwd.dwd_opr_knowledge_chunk_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS dwd.dwd_opr_content_asset_zip (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  asset_no varchar(64) NOT NULL,
  modality_type varchar(16) NOT NULL,
  category_code varchar(64),
  stage_code smallint NOT NULL,
  stage_status smallint NOT NULL,
  start_date date NOT NULL,
  end_date date NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);
COMMENT ON TABLE dwd.dwd_opr_content_asset_zip IS '内容资产拉链表（记录分类、区域、阶段的历史变更）';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.id IS '自增主键';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.asset_no IS '资产编号';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.modality_type IS '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.category_code IS '分类编码';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.stage_code IS '流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.stage_status IS '阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.start_date IS '拉链开始日期（含）';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.end_date IS '拉链结束日期（不含），当前有效记录为 9999-12-31';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.create_time IS '创建时间';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dwd.dwd_opr_content_asset_zip.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_dwd_opr_content_asset_zip_0 ON dwd.dwd_opr_content_asset_zip (tenant_code, asset_no, end_date);
ALTER TABLE dwd.dwd_opr_content_asset_zip ENABLE ROW LEVEL SECURITY;
ALTER TABLE dwd.dwd_opr_content_asset_zip FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dwd.dwd_opr_content_asset_zip;
CREATE POLICY p_tenant_read ON dwd.dwd_opr_content_asset_zip FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dwd.dwd_opr_content_asset_zip;
CREATE POLICY p_tenant_write ON dwd.dwd_opr_content_asset_zip FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS dwd.dwd_cus_user_zip (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  user_code varchar(64) NOT NULL,
  user_type smallint NOT NULL,
  user_status smallint NOT NULL,
  role_code_list varchar(1024),
  start_date date NOT NULL,
  end_date date NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);
COMMENT ON TABLE dwd.dwd_cus_user_zip IS '用户拉链表（记录用户状态、类型、归属区域的历史变更）';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.id IS '自增主键';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.user_code IS '用户编码';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.user_type IS '用户类型：0-平台超级管理员 1-租户管理员 2-管理端普通用户 3-API 账号 4-用户端问答用户（只能登录用户端，0~2 只能登录管理端）';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.user_status IS '用户状态：0-正常 1-锁定 2-停用';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.role_code_list IS '当时拥有的角色编码，逗号分隔';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.start_date IS '拉链开始日期（含）';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.end_date IS '拉链结束日期（不含），当前有效记录为 9999-12-31';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.create_time IS '创建时间';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dwd.dwd_cus_user_zip.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_dwd_cus_user_zip_0 ON dwd.dwd_cus_user_zip (tenant_code, user_code, end_date);
ALTER TABLE dwd.dwd_cus_user_zip ENABLE ROW LEVEL SECURITY;
ALTER TABLE dwd.dwd_cus_user_zip FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dwd.dwd_cus_user_zip;
CREATE POLICY p_tenant_read ON dwd.dwd_cus_user_zip FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dwd.dwd_cus_user_zip;
CREATE POLICY p_tenant_write ON dwd.dwd_cus_user_zip FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

-- ======================================================================
-- dws
-- ======================================================================

CREATE TABLE IF NOT EXISTS dws.dws_opr_content_process_di (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  modality_type varchar(16) NOT NULL,
  upload_qty int NOT NULL DEFAULT 0,
  parse_success_qty int NOT NULL DEFAULT 0,
  parse_fail_qty int NOT NULL DEFAULT 0,
  retry_qty int NOT NULL DEFAULT 0,
  review_qty int NOT NULL DEFAULT 0,
  review_edit_block_qty int NOT NULL DEFAULT 0,
  label_qty int NOT NULL DEFAULT 0,
  store_qty int NOT NULL DEFAULT 0,
  apply_qty int NOT NULL DEFAULT 0,
  avg_parse_ms_qty bigint,
  review_pass_rate decimal(6,4),
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date),
  UNIQUE (tenant_code, region_code, modality_type, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS dws.dws_opr_content_process_di_pdefault PARTITION OF dws.dws_opr_content_process_di DEFAULT;
COMMENT ON TABLE dws.dws_opr_content_process_di IS '内容处理日汇总（租户×区域×模态×日）';
COMMENT ON COLUMN dws.dws_opr_content_process_di.id IS '自增主键';
COMMENT ON COLUMN dws.dws_opr_content_process_di.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dws.dws_opr_content_process_di.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN dws.dws_opr_content_process_di.modality_type IS '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频';
COMMENT ON COLUMN dws.dws_opr_content_process_di.upload_qty IS '上传数';
COMMENT ON COLUMN dws.dws_opr_content_process_di.parse_success_qty IS '解析成功数';
COMMENT ON COLUMN dws.dws_opr_content_process_di.parse_fail_qty IS '解析失败数（重试用尽）';
COMMENT ON COLUMN dws.dws_opr_content_process_di.retry_qty IS '重试次数合计';
COMMENT ON COLUMN dws.dws_opr_content_process_di.review_qty IS '核对完成数';
COMMENT ON COLUMN dws.dws_opr_content_process_di.review_edit_block_qty IS '核对中被修改的内容块数';
COMMENT ON COLUMN dws.dws_opr_content_process_di.label_qty IS '标注条数';
COMMENT ON COLUMN dws.dws_opr_content_process_di.store_qty IS '入库数';
COMMENT ON COLUMN dws.dws_opr_content_process_di.apply_qty IS '应用数';
COMMENT ON COLUMN dws.dws_opr_content_process_di.avg_parse_ms_qty IS '平均解析耗时（毫秒）';
COMMENT ON COLUMN dws.dws_opr_content_process_di.review_pass_rate IS '核对一次通过率（未修改块占比）';
COMMENT ON COLUMN dws.dws_opr_content_process_di.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN dws.dws_opr_content_process_di.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dws.dws_opr_content_process_di.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dws.dws_opr_content_process_di.create_time IS '创建时间';
COMMENT ON COLUMN dws.dws_opr_content_process_di.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dws.dws_opr_content_process_di.update_time IS '最后更新时间';
ALTER TABLE dws.dws_opr_content_process_di ENABLE ROW LEVEL SECURITY;
ALTER TABLE dws.dws_opr_content_process_di FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dws.dws_opr_content_process_di;
CREATE POLICY p_tenant_read ON dws.dws_opr_content_process_di FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dws.dws_opr_content_process_di;
CREATE POLICY p_tenant_write ON dws.dws_opr_content_process_di FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS dws.dws_opr_content_stage_hrs (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  stage_hour_code char(2) NOT NULL,
  stage_code smallint NOT NULL,
  enter_qty int NOT NULL DEFAULT 0,
  finish_qty int NOT NULL DEFAULT 0,
  fail_qty int NOT NULL DEFAULT 0,
  backlog_qty int NOT NULL DEFAULT 0,
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date),
  UNIQUE (tenant_code, region_code, stage_hour_code, stage_code, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS dws.dws_opr_content_stage_hrs_pdefault PARTITION OF dws.dws_opr_content_stage_hrs DEFAULT;
COMMENT ON TABLE dws.dws_opr_content_stage_hrs IS '流水线阶段小时吞吐（租户×阶段×小时，用于队列积压监控）';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.id IS '自增主键';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.stage_hour_code IS '小时，00~23';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.stage_code IS '流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.enter_qty IS '进入该阶段的资产数';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.finish_qty IS '完成该阶段的资产数';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.fail_qty IS '失败数';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.backlog_qty IS '小时末积压数';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.create_time IS '创建时间';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dws.dws_opr_content_stage_hrs.update_time IS '最后更新时间';
ALTER TABLE dws.dws_opr_content_stage_hrs ENABLE ROW LEVEL SECURITY;
ALTER TABLE dws.dws_opr_content_stage_hrs FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dws.dws_opr_content_stage_hrs;
CREATE POLICY p_tenant_read ON dws.dws_opr_content_stage_hrs FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dws.dws_opr_content_stage_hrs;
CREATE POLICY p_tenant_write ON dws.dws_opr_content_stage_hrs FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS dws.dws_fin_tenant_usage_mo (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  stat_month_code char(6) NOT NULL,
  storage_mb_qty bigint NOT NULL DEFAULT 0,
  parse_page_qty bigint NOT NULL DEFAULT 0,
  media_minute_qty bigint NOT NULL DEFAULT 0,
  input_token_qty bigint NOT NULL DEFAULT 0,
  output_token_qty bigint NOT NULL DEFAULT 0,
  cost_amt decimal(18,6),
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date),
  UNIQUE (tenant_code, stat_month_code, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS dws.dws_fin_tenant_usage_mo_pdefault PARTITION OF dws.dws_fin_tenant_usage_mo DEFAULT;
COMMENT ON TABLE dws.dws_fin_tenant_usage_mo IS '租户月度用量汇总（存储、解析页数、音视频时长、智能体 token，用于计费）';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.id IS '自增主键';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.stat_month_code IS '统计月份 YYYYMM';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.storage_mb_qty IS '月末存储量（MB）';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.parse_page_qty IS '解析页数';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.media_minute_qty IS '音视频处理分钟数';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.input_token_qty IS '智能体输入 token';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.output_token_qty IS '智能体输出 token';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.cost_amt IS '估算模型费用（美元）';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.create_time IS '创建时间';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dws.dws_fin_tenant_usage_mo.update_time IS '最后更新时间';
ALTER TABLE dws.dws_fin_tenant_usage_mo ENABLE ROW LEVEL SECURITY;
ALTER TABLE dws.dws_fin_tenant_usage_mo FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dws.dws_fin_tenant_usage_mo;
CREATE POLICY p_tenant_read ON dws.dws_fin_tenant_usage_mo FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dws.dws_fin_tenant_usage_mo;
CREATE POLICY p_tenant_write ON dws.dws_fin_tenant_usage_mo FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

-- ======================================================================
-- ads
-- ======================================================================

CREATE TABLE IF NOT EXISTS ads.ads_opr_content_dashboard_df (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  total_asset_qty bigint NOT NULL DEFAULT 0,
  today_upload_qty int NOT NULL DEFAULT 0,
  pending_review_qty int NOT NULL DEFAULT 0,
  pending_label_qty int NOT NULL DEFAULT 0,
  failed_qty int NOT NULL DEFAULT 0,
  stored_qty bigint NOT NULL DEFAULT 0,
  kb_chunk_qty bigint NOT NULL DEFAULT 0,
  parse_success_rate decimal(6,4),
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date),
  UNIQUE (tenant_code, region_code, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS ads.ads_opr_content_dashboard_df_pdefault PARTITION OF ads.ads_opr_content_dashboard_df DEFAULT;
COMMENT ON TABLE ads.ads_opr_content_dashboard_df IS '内容中心看板指标（天全量快照：截至当日的累计与当日指标）';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.id IS '自增主键';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.total_asset_qty IS '累计资产数';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.today_upload_qty IS '当日上传数';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.pending_review_qty IS '待核对数';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.pending_label_qty IS '待标注数';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.failed_qty IS '失败待处理数';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.stored_qty IS '累计入库数';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.kb_chunk_qty IS '累计知识切片数';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.parse_success_rate IS '当日解析成功率';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.create_time IS '创建时间';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN ads.ads_opr_content_dashboard_df.update_time IS '最后更新时间';
ALTER TABLE ads.ads_opr_content_dashboard_df ENABLE ROW LEVEL SECURITY;
ALTER TABLE ads.ads_opr_content_dashboard_df FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON ads.ads_opr_content_dashboard_df;
CREATE POLICY p_tenant_read ON ads.ads_opr_content_dashboard_df FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON ads.ads_opr_content_dashboard_df;
CREATE POLICY p_tenant_write ON ads.ads_opr_content_dashboard_df FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

CREATE TABLE IF NOT EXISTS ads.ads_opr_content_quality_mo (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  stat_month_code char(6) NOT NULL,
  modality_type varchar(16) NOT NULL,
  category_code varchar(64) NOT NULL,
  asset_qty int NOT NULL DEFAULT 0,
  review_pass_rate decimal(6,4),
  avg_review_minute_qty decimal(10,2),
  auto_classify_accept_rate decimal(6,4),
  travel_date char(8) NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, travel_date),
  UNIQUE (tenant_code, region_code, stat_month_code, modality_type, category_code, travel_date)
) PARTITION BY RANGE (travel_date);
CREATE TABLE IF NOT EXISTS ads.ads_opr_content_quality_mo_pdefault PARTITION OF ads.ads_opr_content_quality_mo DEFAULT;
COMMENT ON TABLE ads.ads_opr_content_quality_mo IS '内容质量月报（模态×分类维度的解析质量与人工工作量）';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.id IS '自增主键';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.stat_month_code IS '统计月份 YYYYMM';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.modality_type IS '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.category_code IS '分类编码';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.asset_qty IS '资产数';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.review_pass_rate IS '核对一次通过率';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.avg_review_minute_qty IS '平均人工核对时长（分钟）';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.auto_classify_accept_rate IS '模型分类建议采纳率';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.travel_date IS '分区日期，格式YYYYMMDD';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.create_time IS '创建时间';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN ads.ads_opr_content_quality_mo.update_time IS '最后更新时间';
ALTER TABLE ads.ads_opr_content_quality_mo ENABLE ROW LEVEL SECURITY;
ALTER TABLE ads.ads_opr_content_quality_mo FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON ads.ads_opr_content_quality_mo;
CREATE POLICY p_tenant_read ON ads.ads_opr_content_quality_mo FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON ads.ads_opr_content_quality_mo;
CREATE POLICY p_tenant_write ON ads.ads_opr_content_quality_mo FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true) AND (current_setting('pdr.region_all', true) = '1' OR region_code = '' OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ','))));

-- ======================================================================
-- dim
-- ======================================================================

CREATE TABLE IF NOT EXISTS dim.dim_cus_region_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  region_code varchar(32) NOT NULL DEFAULT '',
  region_name varchar(128) NOT NULL,
  parent_region_code varchar(32) NOT NULL DEFAULT '',
  region_path varchar(512) NOT NULL,
  region_level smallint NOT NULL,
  region_type varchar(16) NOT NULL DEFAULT 'admin',
  region_sort_qty int,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, region_code)
);
COMMENT ON TABLE dim.dim_cus_region_f IS '区域维度表（树形，区域权限的授权单元）';
COMMENT ON COLUMN dim.dim_cus_region_f.id IS '自增主键';
COMMENT ON COLUMN dim.dim_cus_region_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dim.dim_cus_region_f.region_code IS '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）';
COMMENT ON COLUMN dim.dim_cus_region_f.region_name IS '区域名称';
COMMENT ON COLUMN dim.dim_cus_region_f.parent_region_code IS '上级区域编码，根节点为空串';
COMMENT ON COLUMN dim.dim_cus_region_f.region_path IS '区域路径，形如 /CN/CN-31/CN-31-01/，用于前缀匹配下级';
COMMENT ON COLUMN dim.dim_cus_region_f.region_level IS '区域层级：1-国家 2-省/州 3-城市 4-区县 5-景区/门店';
COMMENT ON COLUMN dim.dim_cus_region_f.region_type IS '区域类型：admin-行政区 scenic-景区 store-门店 virtual-虚拟分组';
COMMENT ON COLUMN dim.dim_cus_region_f.region_sort_qty IS '同级排序号，越小越靠前';
COMMENT ON COLUMN dim.dim_cus_region_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dim.dim_cus_region_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dim.dim_cus_region_f.create_time IS '创建时间';
COMMENT ON COLUMN dim.dim_cus_region_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dim.dim_cus_region_f.update_time IS '最后更新时间';
CREATE INDEX IF NOT EXISTS idx_dim_cus_region_f_0 ON dim.dim_cus_region_f (tenant_code, region_path);
ALTER TABLE dim.dim_cus_region_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE dim.dim_cus_region_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dim.dim_cus_region_f;
CREATE POLICY p_tenant_read ON dim.dim_cus_region_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dim.dim_cus_region_f;
CREATE POLICY p_tenant_write ON dim.dim_cus_region_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS dim.dim_opr_content_modality_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  modality_type varchar(16) NOT NULL,
  modality_name varchar(64) NOT NULL,
  file_ext_list varchar(256) NOT NULL,
  parser_type varchar(32) NOT NULL,
  max_file_size_mb_qty int NOT NULL,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, modality_type)
);
COMMENT ON TABLE dim.dim_opr_content_modality_f IS '内容模态字典（七种模态及其解析通道）';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.id IS '自增主键';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.modality_type IS '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.modality_name IS '模态中文名称';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.file_ext_list IS '允许的文件扩展名，逗号分隔，如 pdf 或 doc,docx';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.parser_type IS '解析通道：mineru-MinerU文档解析 markdown-Markdown直读 media-音视频(ffmpeg+ASR+抽帧经MinerU)';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.max_file_size_mb_qty IS '单文件大小上限（MB）';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.create_time IS '创建时间';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dim.dim_opr_content_modality_f.update_time IS '最后更新时间';
ALTER TABLE dim.dim_opr_content_modality_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE dim.dim_opr_content_modality_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dim.dim_opr_content_modality_f;
CREATE POLICY p_tenant_read ON dim.dim_opr_content_modality_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dim.dim_opr_content_modality_f;
CREATE POLICY p_tenant_write ON dim.dim_opr_content_modality_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS dim.dim_opr_content_stage_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  stage_code smallint NOT NULL,
  stage_name varchar(64) NOT NULL,
  stage_type varchar(16) NOT NULL,
  next_stage_code smallint NOT NULL,
  skip_allowed_status smallint NOT NULL DEFAULT 0,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, stage_code)
);
COMMENT ON TABLE dim.dim_opr_content_stage_f IS '流水线阶段字典';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.id IS '自增主键';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.stage_code IS '流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.stage_name IS '阶段中文名称';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.stage_type IS '执行方式：auto-系统自动 manual-人工 hybrid-系统建议+人工确认';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.next_stage_code IS '默认下一阶段编码，0 表示流程结束';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.skip_allowed_status IS '是否允许跳过：0-不允许 1-允许';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.create_time IS '创建时间';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dim.dim_opr_content_stage_f.update_time IS '最后更新时间';
ALTER TABLE dim.dim_opr_content_stage_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE dim.dim_opr_content_stage_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dim.dim_opr_content_stage_f;
CREATE POLICY p_tenant_read ON dim.dim_opr_content_stage_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dim.dim_opr_content_stage_f;
CREATE POLICY p_tenant_write ON dim.dim_opr_content_stage_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS dim.dim_opr_content_category_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  category_code varchar(64) NOT NULL,
  category_name varchar(128) NOT NULL,
  parent_category_code varchar(64) NOT NULL DEFAULT '',
  category_path varchar(512) NOT NULL,
  target_table_name varchar(128),
  extract_prompt text,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, category_code)
);
COMMENT ON TABLE dim.dim_opr_content_category_f IS '内容分类体系（树形，租户可自定义）';
COMMENT ON COLUMN dim.dim_opr_content_category_f.id IS '自增主键';
COMMENT ON COLUMN dim.dim_opr_content_category_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dim.dim_opr_content_category_f.category_code IS '分类编码';
COMMENT ON COLUMN dim.dim_opr_content_category_f.category_name IS '分类名称';
COMMENT ON COLUMN dim.dim_opr_content_category_f.parent_category_code IS '上级分类编码，根为空串';
COMMENT ON COLUMN dim.dim_opr_content_category_f.category_path IS '分类路径，形如 /contract/supplier/';
COMMENT ON COLUMN dim.dim_opr_content_category_f.target_table_name IS '入库回填目标表（库名.表名），为空表示只进知识库，如 ods.ods_fin_invoice_extract_di';
COMMENT ON COLUMN dim.dim_opr_content_category_f.extract_prompt IS '回填字段抽取提示词（交给智能体辅助模型）';
COMMENT ON COLUMN dim.dim_opr_content_category_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dim.dim_opr_content_category_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dim.dim_opr_content_category_f.create_time IS '创建时间';
COMMENT ON COLUMN dim.dim_opr_content_category_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dim.dim_opr_content_category_f.update_time IS '最后更新时间';
ALTER TABLE dim.dim_opr_content_category_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE dim.dim_opr_content_category_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dim.dim_opr_content_category_f;
CREATE POLICY p_tenant_read ON dim.dim_opr_content_category_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dim.dim_opr_content_category_f;
CREATE POLICY p_tenant_write ON dim.dim_opr_content_category_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));

CREATE TABLE IF NOT EXISTS dim.dim_opr_label_schema_f (
  id bigserial NOT NULL,
  tenant_code varchar(32) NOT NULL,
  label_code varchar(64) NOT NULL,
  label_name varchar(128) NOT NULL,
  label_type smallint NOT NULL,
  label_color varchar(16),
  value_schema jsonb,
  record_status smallint NOT NULL DEFAULT 0,
  create_user_code varchar(64) NOT NULL DEFAULT 'system',
  create_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_user_code varchar(64) NOT NULL DEFAULT 'system',
  update_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE (tenant_code, label_code)
);
COMMENT ON TABLE dim.dim_opr_label_schema_f IS '标注体系定义（标签及其可用的标注形态）';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.id IS '自增主键';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.tenant_code IS '租户编码（SaaS 租户隔离）';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.label_code IS '标签编码';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.label_name IS '标签名称';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.label_type IS '标注形态：1-文本实体 2-图像区域框 3-音视频时间片 4-整篇标签 5-表格单元格';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.label_color IS '前端展示颜色，如 #409EFF';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.value_schema IS '标签取值约束（JSON Schema），为空表示自由文本';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.record_status IS '记录状态：0-正常 1-已删除（逻辑删除）';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.create_user_code IS '创建人用户编码，系统任务为 system';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.create_time IS '创建时间';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.update_user_code IS '最后更新人用户编码';
COMMENT ON COLUMN dim.dim_opr_label_schema_f.update_time IS '最后更新时间';
ALTER TABLE dim.dim_opr_label_schema_f ENABLE ROW LEVEL SECURITY;
ALTER TABLE dim.dim_opr_label_schema_f FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_tenant_read ON dim.dim_opr_label_schema_f;
CREATE POLICY p_tenant_read ON dim.dim_opr_label_schema_f FOR SELECT USING ((tenant_code = current_setting('pdr.tenant_code', true)) OR tenant_code = 'platform');
DROP POLICY IF EXISTS p_tenant_write ON dim.dim_opr_label_schema_f;
CREATE POLICY p_tenant_write ON dim.dim_opr_label_schema_f FOR ALL USING (tenant_code = current_setting('pdr.tenant_code', true)) WITH CHECK (tenant_code = current_setting('pdr.tenant_code', true));
