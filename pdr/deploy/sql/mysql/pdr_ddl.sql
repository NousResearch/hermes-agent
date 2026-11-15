-- 由 server/pdr_server/db/tables.py 生成，请勿手改：python -m pdr_server.db.render
CREATE DATABASE IF NOT EXISTS src DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE DATABASE IF NOT EXISTS ods DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE DATABASE IF NOT EXISTS dwd DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE DATABASE IF NOT EXISTS dws DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE DATABASE IF NOT EXISTS ads DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE DATABASE IF NOT EXISTS dim DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_0900_ai_ci;

-- ======================================================================
-- src
-- ======================================================================

CREATE TABLE IF NOT EXISTS src.src_cus_tenant_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `tenant_name` varchar(128) NOT NULL COMMENT '租户名称',
  `tenant_type` smallint NOT NULL COMMENT '租户类型：0-平台运营方 1-企业租户 2-试用租户',
  `tenant_status` smallint NOT NULL COMMENT '租户状态：0-正常 1-停用 2-已过期',
  `agent_profile_name` varchar(64) NOT NULL COMMENT '对应 hermes 智能体 profile 名称，形如 t_<tenant_code>',
  `expire_date` date COMMENT '服务到期日期',
  `max_user_qty` int COMMENT '最大用户数配额',
  `max_storage_gb_qty` int COMMENT '存储配额（GB）',
  `contact_name` varchar(64) COMMENT '联系人姓名',
  `contact_phone_no` varchar(32) COMMENT '联系人电话号码',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_tenant_f_0 (tenant_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='租户表（SaaS 租户主数据）';

CREATE TABLE IF NOT EXISTS src.src_cus_user_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `user_code` varchar(64) NOT NULL COMMENT '用户编码（租户内唯一，登录名）',
  `user_name` varchar(64) NOT NULL COMMENT '用户姓名',
  `password_hash` varchar(256) NOT NULL COMMENT '密码哈希（argon2id）',
  `email_addr` varchar(128) COMMENT '电子邮箱地址',
  `mobile_no` varchar(32) COMMENT '手机号码',
  `user_type` smallint NOT NULL COMMENT '用户类型：0-平台超级管理员 1-租户管理员 2-普通用户 3-API 账号',
  `user_status` smallint NOT NULL COMMENT '用户状态：0-正常 1-锁定 2-停用',
  `last_login_time` datetime(3) COMMENT '最近登录时间',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_user_f_0 (tenant_code, user_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';

CREATE TABLE IF NOT EXISTS src.src_cus_role_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `role_code` varchar(64) NOT NULL COMMENT '角色编码',
  `role_name` varchar(64) NOT NULL COMMENT '角色名称',
  `role_type` smallint NOT NULL COMMENT '角色类型：0-系统内置（不可删） 1-租户自定义',
  `role_desc` varchar(512) COMMENT '角色说明',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_role_f_0 (tenant_code, role_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='角色表';

CREATE TABLE IF NOT EXISTS src.src_cus_permission_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `permission_code` varchar(128) NOT NULL COMMENT '权限点编码，形如 content:review:submit',
  `permission_name` varchar(128) NOT NULL COMMENT '权限点名称',
  `permission_type` smallint NOT NULL COMMENT '权限点类型：1-菜单 2-按钮 3-接口',
  `parent_permission_code` varchar(128) NOT NULL DEFAULT '' COMMENT '上级权限点编码',
  `api_method_name` varchar(16) COMMENT '接口 HTTP 方法（类型为接口时）：GET/POST/PUT/PATCH/DELETE',
  `api_path_name` varchar(256) COMMENT '接口路径模板（类型为接口时），如 /api/content/assets/{id}',
  `route_path_name` varchar(256) COMMENT '前端路由路径（类型为菜单时）',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_permission_f_0 (tenant_code, permission_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='权限点表（菜单/按钮/接口，全平台共享，tenant_code=platform）';

CREATE TABLE IF NOT EXISTS src.src_cus_user_role_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `user_code` varchar(64) NOT NULL COMMENT '用户编码',
  `role_code` varchar(64) NOT NULL COMMENT '角色编码',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_user_role_f_0 (tenant_code, user_code, role_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户-角色关系表';

CREATE TABLE IF NOT EXISTS src.src_cus_role_permission_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `role_code` varchar(64) NOT NULL COMMENT '角色编码',
  `permission_code` varchar(128) NOT NULL COMMENT '权限点编码',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_role_permission_f_0 (tenant_code, role_code, permission_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='角色-权限点关系表';

CREATE TABLE IF NOT EXISTS src.src_cus_user_region_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `user_code` varchar(64) NOT NULL COMMENT '用户编码',
  `primary_status` smallint NOT NULL DEFAULT 0 COMMENT '是否主区域：0-否 1-是',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_user_region_f_0 (tenant_code, user_code, region_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户归属区域表（用户可归属多个区域，作为“本区域”取值来源）';

CREATE TABLE IF NOT EXISTS src.src_cus_role_data_scope_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `role_code` varchar(64) NOT NULL COMMENT '角色编码',
  `resource_type` varchar(64) NOT NULL COMMENT '资源类型：content_asset-内容资产 knowledge-知识库 business_extract-业务回填数据 report-报表 *-全部',
  `scope_type` smallint NOT NULL COMMENT '范围类型：1-本租户全部 2-本区域及下级 3-仅本区域 4-自定义区域 5-仅本人创建',
  `action_type` varchar(16) NOT NULL DEFAULT '*' COMMENT '适用操作：read-读 write-写 *-读写',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_role_data_scope_f_0 (tenant_code, role_code, resource_type, action_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='角色数据范围表（行级权限：按资源类型配置可见范围）';

CREATE TABLE IF NOT EXISTS src.src_cus_role_region_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `role_code` varchar(64) NOT NULL COMMENT '角色编码',
  `resource_type` varchar(64) NOT NULL COMMENT '资源类型：content_asset-内容资产 knowledge-知识库 business_extract-业务回填数据 report-报表 *-全部',
  `include_child_status` smallint NOT NULL DEFAULT 1 COMMENT '是否包含下级区域：0-否 1-是',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_role_region_f_0 (tenant_code, role_code, resource_type, region_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='角色自定义区域表（数据范围为 4-自定义区域 时生效）';

CREATE TABLE IF NOT EXISTS src.src_cus_role_column_perm_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `role_code` varchar(64) NOT NULL COMMENT '角色编码',
  `table_name` varchar(128) NOT NULL COMMENT '受控表（库名.表名），如 ods.ods_fin_invoice_extract_di',
  `column_name` varchar(128) NOT NULL COMMENT '受控字段名',
  `perm_type` smallint NOT NULL COMMENT '列权限：0-不可见 1-脱敏可见 2-只读 3-可读写',
  `mask_rule_type` varchar(32) COMMENT '脱敏规则：mobile-手机号 id_card-证件号 amount-金额区间 name-姓名 full-全遮蔽',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_role_column_perm_f_0 (tenant_code, role_code, table_name, column_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='角色列级权限表（未配置的列默认可读写）';

CREATE TABLE IF NOT EXISTS src.src_cus_api_token_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `user_code` varchar(64) NOT NULL COMMENT '所属用户编码',
  `token_name` varchar(64) NOT NULL COMMENT '令牌名称',
  `token_hash` varchar(128) NOT NULL COMMENT '令牌 SHA-256 哈希（明文只在创建时展示一次）',
  `token_prefix_code` varchar(16) NOT NULL COMMENT '令牌前缀（用于界面识别）',
  `expire_time` datetime(3) COMMENT '过期时间，为空表示不过期',
  `token_status` smallint NOT NULL COMMENT '令牌状态：0-有效 1-已吊销',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_cus_api_token_f_0 (token_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='API 访问令牌表（供 API 账号与外部系统调用）';

CREATE TABLE IF NOT EXISTS src.src_cus_operate_log_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `user_code` varchar(64) NOT NULL COMMENT '操作人用户编码',
  `operate_type` varchar(64) NOT NULL COMMENT '操作类型，开放取值，与权限点编码一致，如 config:update / content:review:submit / role:grant',
  `resource_type` varchar(64) COMMENT '操作对象资源类型：content_asset-内容资产 knowledge-知识库 config-配置 user-用户 role-角色 tenant-租户 business_extract-业务回填数据',
  `resource_no` varchar(128) COMMENT '操作对象编号',
  `request_path_name` varchar(256) COMMENT '请求接口路径',
  `client_ip_code` varchar(64) COMMENT '客户端 IP',
  `operate_status` smallint NOT NULL COMMENT '操作结果：0-成功 1-失败 2-拒绝（无权限）',
  `detail_json` json COMMENT '操作详情（变更前后值等，敏感字段已脱敏）',
  `operate_time` datetime(3) NOT NULL COMMENT '操作时间',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  KEY idx_src_cus_operate_log_di_0 (tenant_code, user_code, operate_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='操作审计日志（天增量）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS src.src_opr_config_item_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `config_key` varchar(256) NOT NULL COMMENT '配置键（点分路径），如 hermes.agent.max_turns / pdr.pipeline.retry.max_attempt_qty',
  `config_domain_type` varchar(16) NOT NULL COMMENT '配置归属：pdr-PDR系统配置 hermes-智能体配置',
  `group_code` varchar(64) NOT NULL COMMENT '页面分组编码（取配置键第一段，如 agent / compression）',
  `group_name` varchar(64) NOT NULL COMMENT '页面分组中文名',
  `config_name` varchar(128) NOT NULL COMMENT '配置项中文名称',
  `config_desc` longtext COMMENT '配置项中文说明',
  `value_type` varchar(16) NOT NULL COMMENT '取值类型：bool-布尔 int-整数 float-小数 str-字符串 enum-枚举 list-列表 map-键值对 secret-密钥引用',
  `default_value` json COMMENT '默认值（JSON 编码）',
  `value_range` json COMMENT '取值范围：{min,max} 或 {enum:[{value,label}]} 或 {pattern}',
  `reload_type` smallint NOT NULL COMMENT '生效方式：0-立即热加载 1-下一会话生效（保护提示缓存） 2-需重启服务',
  `scope_type` smallint NOT NULL COMMENT '可配置层级：0-仅平台 1-平台与租户均可',
  `sort_qty` int COMMENT '组内排序号',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_config_item_f_0 (tenant_code, config_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='配置项字典（hermes 智能体与 PDR 系统配置项的中文元数据）';

CREATE TABLE IF NOT EXISTS src.src_opr_config_value_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `config_key` varchar(256) NOT NULL COMMENT '配置键，关联 src.src_opr_config_item_f.config_key',
  `config_value` json COMMENT '覆盖值（JSON 编码）',
  `value_version_qty` int NOT NULL DEFAULT 1 COMMENT '版本号，每次保存 +1，用于乐观锁',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_config_value_f_0 (tenant_code, config_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='租户配置覆盖值（平台级值以 config.yaml 为准，本表只存租户覆盖）';

CREATE TABLE IF NOT EXISTS src.src_opr_config_change_log_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `config_key` varchar(256) NOT NULL COMMENT '配置键',
  `change_scope_type` smallint NOT NULL COMMENT '变更层级：0-平台（写入 config.yaml） 1-租户（写入覆盖表）',
  `before_value` json COMMENT '变更前值（密钥类只记录是否设置）',
  `after_value` json COMMENT '变更后值（密钥类只记录是否设置）',
  `apply_status` smallint NOT NULL COMMENT '生效状态：0-已生效 1-待下一会话 2-待重启 3-生效失败',
  `config_file_sha256_code` char(64) COMMENT '写入后 config.yaml 的 SHA-256，便于核对页面与文件一致',
  `change_time` datetime(3) NOT NULL COMMENT '变更时间',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  KEY idx_src_opr_config_change_log_di_0 (tenant_code, config_key, change_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='配置变更记录（平台级与租户级修改均记录，天增量）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS src.src_opr_content_batch_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `batch_no` varchar(64) NOT NULL COMMENT '批次编号',
  `batch_name` varchar(128) NOT NULL COMMENT '批次名称',
  `file_qty` int NOT NULL DEFAULT 0 COMMENT '批次文件数',
  `batch_status` smallint NOT NULL COMMENT '批次状态：0-上传中 1-已提交 2-处理中 3-全部完成 4-部分失败',
  `default_category_code` varchar(64) COMMENT '批次默认分类编码',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_content_batch_f_0 (tenant_code, batch_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容上传批次表';

CREATE TABLE IF NOT EXISTS src.src_opr_content_asset_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `batch_no` varchar(64) COMMENT '所属批次编号',
  `asset_name` varchar(256) NOT NULL COMMENT '资产名称（默认取原文件名）',
  `modality_type` varchar(16) NOT NULL COMMENT '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频',
  `file_ext_code` varchar(16) NOT NULL COMMENT '文件扩展名，如 pdf / docx / mp4',
  `file_size_qty` bigint NOT NULL COMMENT '文件大小（字节）',
  `file_sha256_code` char(64) NOT NULL COMMENT '文件 SHA-256（租户内去重）',
  `storage_uri_name` varchar(512) NOT NULL COMMENT '原件存储地址，如 file:///data/... 或 s3://bucket/key',
  `page_qty` int COMMENT '页数/工作表数（文档类）',
  `duration_ms_qty` bigint COMMENT '时长（毫秒，音视频类）',
  `stage_code` smallint NOT NULL COMMENT '当前流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用',
  `stage_status` smallint NOT NULL COMMENT '当前阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消',
  `category_code` varchar(64) COMMENT '确认后的分类编码',
  `assignee_user_code` varchar(64) COMMENT '当前人工阶段处理人用户编码',
  `parse_backend_name` varchar(64) COMMENT '解析所用后端，如 mineru:standard / markdown / media',
  `business_date` date COMMENT '业务日期（如发票日期、合同签署日期），可由抽取回填',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_content_asset_f_0 (tenant_code, asset_no),
  KEY idx_src_opr_content_asset_f_0 (tenant_code, region_code, stage_code, stage_status),
  KEY idx_src_opr_content_asset_f_1 (tenant_code, file_sha256_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容资产主表（一个上传文件一行，记录当前阶段与状态）';

CREATE TABLE IF NOT EXISTS src.src_opr_content_object_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '所属资产编号',
  `object_type` varchar(32) NOT NULL COMMENT '派生类型：page_image-页面图 crop_image-版面裁切图 frame-视频关键帧 audio_track-音轨 subtitle_srt-SRT字幕 subtitle_vtt-VTT字幕 mineru_raw-MinerU原始产物包 export-导出包',
  `storage_uri_name` varchar(512) NOT NULL COMMENT '存储地址',
  `mime_type` varchar(128) COMMENT 'MIME 类型，开放取值，遵循 IANA 媒体类型，如 application/pdf',
  `file_size_qty` bigint COMMENT '文件大小（字节）',
  `page_no` int COMMENT '关联页码（从 1 开始）',
  `time_ms_qty` bigint COMMENT '关联时间点（毫秒，关键帧用）',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  KEY idx_src_opr_content_object_f_0 (tenant_code, asset_no, object_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容派生文件表（转换 PDF、页面图、裁切图、关键帧、音轨、字幕、导出包等）';

CREATE TABLE IF NOT EXISTS src.src_opr_content_media_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `duration_ms_qty` bigint NOT NULL COMMENT '时长（毫秒）',
  `container_name` varchar(32) COMMENT '封装格式，如 mp4 / mkv / mp3 / wav',
  `video_codec_name` varchar(32) COMMENT '视频编码，如 h264',
  `audio_codec_name` varchar(32) COMMENT '音频编码，如 aac',
  `width_qty` int COMMENT '画面宽度（像素）',
  `height_qty` int COMMENT '画面高度（像素）',
  `frame_per_sec_qty` decimal(8,3) COMMENT '帧率（帧/秒）',
  `sample_rate_qty` int COMMENT '音频采样率（Hz）',
  `channel_qty` smallint COMMENT '音频声道数',
  `speech_lang_code` varchar(16) COMMENT '识别出的语种，如 zh / en',
  `frame_qty` int COMMENT '抽取关键帧数',
  `segment_qty` int COMMENT '转写语音片段数',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_content_media_f_0 (tenant_code, asset_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='音视频元数据表（ffprobe 抽取）';

CREATE TABLE IF NOT EXISTS src.src_opr_content_stage_task_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `task_no` varchar(64) NOT NULL COMMENT '任务编号',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `stage_code` smallint NOT NULL COMMENT '流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用',
  `task_status` smallint NOT NULL COMMENT '任务阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消',
  `attempt_qty` int NOT NULL DEFAULT 0 COMMENT '已执行次数',
  `max_attempt_qty` int NOT NULL DEFAULT 3 COMMENT '最大执行次数（来自 config.yaml pipeline.retry）',
  `next_retry_time` datetime(3) COMMENT '下次重试时间（指数退避）',
  `queue_message_code` varchar(64) COMMENT 'Redis Stream 消息 ID',
  `worker_name` varchar(128) COMMENT '执行该任务的 worker 名称',
  `operator_user_code` varchar(64) COMMENT '人工阶段的操作人用户编码；自动阶段为 system',
  `start_time` datetime(3) COMMENT '开始时间',
  `end_time` datetime(3) COMMENT '结束时间',
  `error_code` varchar(64) COMMENT '失败错误码，如 MINERU_TIMEOUT / ASR_FAILED',
  `error_msg` longtext COMMENT '失败详情',
  `task_param_json` json COMMENT '任务参数（解析档位、OCR 语言等）',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_content_stage_task_f_0 (tenant_code, task_no),
  KEY idx_src_opr_content_stage_task_f_0 (tenant_code, asset_no, stage_code),
  KEY idx_src_opr_content_stage_task_f_1 (task_status, next_retry_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='流水线阶段任务表（每个资产每个阶段一次执行一行，含重试）';

CREATE TABLE IF NOT EXISTS src.src_opr_content_stage_log_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `task_no` varchar(64) COMMENT '关联任务编号',
  `from_stage_code` smallint NOT NULL COMMENT '变更前流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用，0-新建',
  `from_stage_status` smallint NOT NULL COMMENT '变更前阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消',
  `to_stage_code` smallint NOT NULL COMMENT '变更后流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用',
  `to_stage_status` smallint NOT NULL COMMENT '变更后阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消',
  `operator_user_code` varchar(64) NOT NULL COMMENT '操作人用户编码，系统自动为 system',
  `change_reason` varchar(512) COMMENT '变更原因（驳回意见、失败原因、跳过说明）',
  `change_time` datetime(3) NOT NULL COMMENT '变更时间',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  KEY idx_src_opr_content_stage_log_di_0 (tenant_code, asset_no, change_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='资产状态流转日志（每次状态变化一行，天增量）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS src.src_opr_content_block_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `block_no` varchar(64) NOT NULL COMMENT '内容块编号',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `block_type` varchar(16) NOT NULL COMMENT '内容块类型：title-标题 text-正文 list-列表 table-表格 image-图片 equation-公式 code-代码 header-页眉 footer-页脚 sheet-工作表 frame-视频关键帧 speech-语音片段',
  `block_order_qty` int NOT NULL COMMENT '阅读顺序号（资产内从 1 开始）',
  `parent_block_no` varchar(64) COMMENT '父块编号（表格标题/图片说明挂在表格/图片上）',
  `page_no` int COMMENT '页码/工作表序号（从 1 开始）',
  `bbox_json` json COMMENT '版面坐标 [x0,y0,x1,y1]（页面归一化到 0~1000）',
  `title_level_qty` smallint COMMENT '标题层级（block_type=title 时 1~6）',
  `start_ms_qty` bigint COMMENT '开始时间（毫秒，speech/frame 用）',
  `end_ms_qty` bigint COMMENT '结束时间（毫秒，speech 用）',
  `speaker_code` varchar(32) COMMENT '说话人标识（语音片段，若开启说话人分离）',
  `origin_text` longtext COMMENT '解析原始文本（不可改，便于对比）',
  `review_text` longtext COMMENT '人工核对后文本，为空表示与原文一致',
  `table_html` longtext COMMENT '表格 HTML（block_type=table/sheet）',
  `table_json` json COMMENT '表格结构化数据（二维数组，含合并单元格信息）',
  `image_object_id` bigint COMMENT '关联图片派生文件 id（src.src_opr_content_object_f.id）',
  `confidence_rate` decimal(5,4) COMMENT '解析/识别置信度 0~1',
  `review_status` smallint NOT NULL DEFAULT 0 COMMENT '核对状态：0-未核对 1-已确认无误 2-已修改 3-标记删除',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_content_block_f_0 (tenant_code, block_no),
  KEY idx_src_opr_content_block_f_0 (tenant_code, asset_no, block_order_qty)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容块工作表（解析结果的结构化内容，供人工核对、标注；入库后同步到 DWD）';

CREATE TABLE IF NOT EXISTS src.src_opr_media_align_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `frame_block_no` varchar(64) NOT NULL COMMENT '关键帧内容块编号（block_type=frame）',
  `speech_block_no` varchar(64) NOT NULL COMMENT '语音片段内容块编号（block_type=speech）',
  `overlap_ms_qty` bigint NOT NULL COMMENT '两者时间重叠时长（毫秒）',
  `align_type` smallint NOT NULL COMMENT '对齐方式：1-时间重叠 2-最近邻（帧落在片段间隙时） 3-人工调整',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_media_align_f_0 (tenant_code, frame_block_no, speech_block_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='音视频对齐表（关键帧与语音片段按时间重叠对齐，用于图文字幕联合检索）';

CREATE TABLE IF NOT EXISTS src.src_opr_content_label_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `label_no` varchar(64) NOT NULL COMMENT '标注编号',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `block_no` varchar(64) COMMENT '内容块编号，整篇标签为空',
  `label_code` varchar(64) NOT NULL COMMENT '标签编码（dim.dim_opr_label_schema_f）',
  `label_type` smallint NOT NULL COMMENT '标注形态：1-文本实体 2-图像区域框 3-音视频时间片 4-整篇标签 5-表格单元格',
  `label_value` longtext COMMENT '标注值（实体文本或用户填写值）',
  `char_start_qty` int COMMENT '文本实体起始字符偏移',
  `char_end_qty` int COMMENT '文本实体结束字符偏移（不含）',
  `bbox_json` json COMMENT '图像区域框 [x0,y0,x1,y1]',
  `start_ms_qty` bigint COMMENT '时间片开始（毫秒）',
  `end_ms_qty` bigint COMMENT '时间片结束（毫秒）',
  `cell_json` json COMMENT '表格单元格坐标 {row,col}',
  `label_source_type` smallint NOT NULL COMMENT '标注来源：0-人工 1-模型预标注 2-模型预标注经人工确认',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_content_label_f_0 (tenant_code, label_no),
  KEY idx_src_opr_content_label_f_0 (tenant_code, asset_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='标注记录表';

CREATE TABLE IF NOT EXISTS src.src_opr_content_category_rel_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `category_code` varchar(64) NOT NULL COMMENT '分类编码',
  `classify_source_type` smallint NOT NULL COMMENT '分类来源：0-人工 1-规则 2-模型建议',
  `confidence_rate` decimal(5,4) COMMENT '模型建议置信度 0~1',
  `confirm_status` smallint NOT NULL COMMENT '确认状态：0-待确认 1-已采纳 2-已否决',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_content_category_rel_f_0 (tenant_code, asset_no, category_code, classify_source_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='资产分类关系表（含模型建议与人工确认）';

CREATE TABLE IF NOT EXISTS src.src_opr_knowledge_base_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `kb_code` varchar(64) NOT NULL COMMENT '知识库编码',
  `kb_name` varchar(128) NOT NULL COMMENT '知识库名称',
  `kb_desc` varchar(512) COMMENT '知识库说明（提供给智能体的工具描述）',
  `embedding_model_name` varchar(128) NOT NULL COMMENT '向量模型名称',
  `embedding_dim_qty` int NOT NULL COMMENT '向量维度',
  `kb_status` smallint NOT NULL COMMENT '知识库状态：0-启用 1-停用',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_knowledge_base_f_0 (tenant_code, kb_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库表（入库内容的应用载体，挂载到租户智能体）';

CREATE TABLE IF NOT EXISTS src.src_opr_content_application_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `apply_type` smallint NOT NULL COMMENT '应用类型：1-知识库检索 2-业务表回填 3-导出下载 4-智能体会话附件',
  `target_code` varchar(256) NOT NULL COMMENT '应用目标：知识库编码 / 回填表（库名.表名） / 导出格式',
  `apply_status` smallint NOT NULL COMMENT '应用状态：0-待执行 1-执行中 2-成功 3-失败 4-已撤回',
  `apply_result_json` json COMMENT '执行结果（切片数、回填行数、导出地址等）',
  `apply_time` datetime(3) COMMENT '执行完成时间',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_src_opr_content_application_f_0 (tenant_code, asset_no, apply_type, target_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容应用记录表（入库内容被投放到哪些应用）';

-- ======================================================================
-- ods
-- ======================================================================

CREATE TABLE IF NOT EXISTS ods.ods_opr_mineru_parse_result_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `task_no` varchar(64) NOT NULL COMMENT '解析任务编号',
  `parser_type` varchar(32) NOT NULL COMMENT '解析通道：mineru-MinerU文档解析 markdown-Markdown直读 media-音视频管线',
  `mineru_version_name` varchar(32) COMMENT 'MinerU 版本号',
  `mineru_tier_name` varchar(16) COMMENT 'MinerU 档位：flash / basic / standard / advanced',
  `mineru_task_code` varchar(64) COMMENT 'MinerU 服务侧任务 ID',
  `middle_json_uri_name` varchar(512) COMMENT 'middle_json.json 存储地址',
  `markdown_uri_name` varchar(512) COMMENT 'markdown.md 存储地址',
  `structured_json_uri_name` varchar(512) COMMENT 'structured_content.json 存储地址',
  `block_qty` int COMMENT '解析出的内容块数',
  `parse_duration_ms_qty` bigint COMMENT '解析耗时（毫秒）',
  `parse_status` smallint NOT NULL COMMENT '解析结果：0-成功 1-失败 2-部分成功',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  KEY idx_ods_opr_mineru_parse_result_di_0 (tenant_code, asset_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='MinerU 原始解析结果（每次解析一行，分区日期=解析日期）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS ods.ods_opr_media_transcript_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `task_no` varchar(64) NOT NULL COMMENT '转写任务编号',
  `segment_order_qty` int NOT NULL COMMENT '片段顺序号',
  `start_ms_qty` bigint NOT NULL COMMENT '开始时间（毫秒）',
  `end_ms_qty` bigint NOT NULL COMMENT '结束时间（毫秒）',
  `speaker_code` varchar(32) COMMENT '说话人标识',
  `segment_text` longtext NOT NULL COMMENT '转写文本',
  `word_json` json COMMENT '词级时间戳 [{word,start_ms,end_ms,prob}]',
  `avg_logprob_score` decimal(8,4) COMMENT '平均对数概率（ASR 置信参考，越接近 0 越可信）',
  `asr_model_name` varchar(64) COMMENT 'ASR 模型名称，如 faster-whisper-large-v3',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  KEY idx_ods_opr_media_transcript_di_0 (tenant_code, asset_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='语音转写原始片段（ASR 原样输出，分区日期=转写日期）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS ods.ods_opr_media_frame_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `task_no` varchar(64) NOT NULL COMMENT '抽帧任务编号',
  `frame_order_qty` int NOT NULL COMMENT '帧顺序号',
  `time_ms_qty` bigint NOT NULL COMMENT '帧时间点（毫秒）',
  `extract_type` smallint NOT NULL COMMENT '抽帧方式：1-场景切换 2-固定间隔 3-人工补帧',
  `scene_change_score` decimal(6,4) COMMENT '场景切换分值 0~1',
  `storage_uri_name` varchar(512) NOT NULL COMMENT '帧图片存储地址',
  `ocr_text` longtext COMMENT '帧图片经 MinerU 识别的文字',
  `caption_text` longtext COMMENT '视觉模型生成的画面描述（可选）',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  KEY idx_ods_opr_media_frame_di_0 (tenant_code, asset_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='视频抽帧原始记录（分区日期=抽帧日期）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS ods.ods_opr_agent_session_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `session_code` varchar(128) NOT NULL COMMENT 'hermes 会话 ID',
  `user_code` varchar(64) NOT NULL COMMENT '发起用户编码',
  `model_name` varchar(128) COMMENT '使用的模型',
  `provider_name` varchar(64) COMMENT '模型提供商',
  `input_token_qty` bigint NOT NULL DEFAULT 0 COMMENT '输入 token 数',
  `output_token_qty` bigint NOT NULL DEFAULT 0 COMMENT '输出 token 数',
  `cache_read_token_qty` bigint NOT NULL DEFAULT 0 COMMENT '缓存命中 token 数',
  `cost_amt` decimal(18,6) COMMENT '估算费用（美元）',
  `session_start_time` datetime(3) COMMENT '会话开始时间',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  UNIQUE KEY uk_ods_opr_agent_session_di_0 (tenant_code, session_code, travel_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='智能体会话用量原始记录（从各租户 hermes profile 同步，分区日期=会话日期）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS ods.ods_fin_invoice_extract_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '来源资产编号',
  `invoice_no` varchar(64) COMMENT '发票号码',
  `invoice_code` varchar(64) COMMENT '发票代码',
  `invoice_type` varchar(32) COMMENT '发票类型：vat_special-增值税专票 vat_normal-增值税普票 e_invoice-全电发票 itinerary-航空行程单 other-其他',
  `invoice_date` date COMMENT '开票日期',
  `seller_name` varchar(256) COMMENT '销售方名称',
  `buyer_name` varchar(256) COMMENT '购买方名称',
  `invoice_amt` decimal(18,2) COMMENT '价税合计金额',
  `tax_amt` decimal(18,2) COMMENT '税额',
  `tax_rate` decimal(6,4) COMMENT '税率',
  `verify_status` smallint NOT NULL DEFAULT 0 COMMENT '人工核验状态：0-待核验 1-已核验 2-有误',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  KEY idx_ods_fin_invoice_extract_di_0 (tenant_code, invoice_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='发票抽取原始数据（发票类内容入库回填，分区日期=开票日期）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS ods.ods_sup_contract_extract_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '来源资产编号',
  `contract_no` varchar(64) COMMENT '合同编号',
  `sup_name` varchar(256) COMMENT '供应商名称',
  `sup_type` varchar(32) COMMENT '供应商类型：hotel-酒店 airline-航司 ground-地接 attraction-景点 other-其他',
  `sign_date` date COMMENT '签署日期',
  `start_date` date COMMENT '合作开始日期',
  `end_date` date COMMENT '合作结束日期',
  `settlement_type` varchar(32) COMMENT '结算方式：prepay-预付 monthly-月结 per_order-单结',
  `net_price_amt` decimal(18,2) COMMENT '结算价金额（合同主报价）',
  `commission_rate` decimal(6,4) COMMENT '佣金比例',
  `verify_status` smallint NOT NULL DEFAULT 0 COMMENT '人工核验状态：0-待核验 1-已核验 2-有误',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  KEY idx_ods_sup_contract_extract_di_0 (tenant_code, contract_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='供应商合同抽取原始数据（分区日期=签署日期）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS ods.ods_prd_product_extract_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '来源资产编号',
  `prd_name` varchar(256) COMMENT '产品名称',
  `prd_type` varchar(32) COMMENT '产品类型：flight-机票 hotel-酒店 tour-跟团游 combo-组合产品 attraction-景点门票',
  `attraction_name` varchar(256) COMMENT '主要景点名称',
  `tour_day_qty` int COMMENT '行程天数',
  `adult_price_amt` decimal(18,2) COMMENT '成人价',
  `child_price_amt` decimal(18,2) COMMENT '儿童价',
  `valid_start_date` date COMMENT '有效期开始日期',
  `valid_end_date` date COMMENT '有效期结束日期',
  `verify_status` smallint NOT NULL DEFAULT 0 COMMENT '人工核验状态：0-待核验 1-已核验 2-有误',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='产品资料抽取原始数据（线路/酒店/门票宣传资料，分区日期=资料生效日期）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

-- ======================================================================
-- dwd
-- ======================================================================

CREATE TABLE IF NOT EXISTS dwd.dwd_opr_content_block_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `block_no` varchar(64) NOT NULL COMMENT '内容块编号',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `modality_type` varchar(16) NOT NULL COMMENT '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频',
  `category_code` varchar(64) NOT NULL COMMENT '资产分类编码',
  `block_type` varchar(16) NOT NULL COMMENT '内容块类型：title-标题 text-正文 list-列表 table-表格 image-图片 equation-公式 code-代码 header-页眉 footer-页脚 sheet-工作表 frame-视频关键帧 speech-语音片段',
  `block_order_qty` int NOT NULL COMMENT '阅读顺序号',
  `page_no` int COMMENT '页码',
  `start_ms_qty` bigint COMMENT '开始时间（毫秒）',
  `end_ms_qty` bigint COMMENT '结束时间（毫秒）',
  `final_text` longtext COMMENT '最终文本（核对文本优先，否则原始文本）',
  `table_json` json COMMENT '表格结构化数据',
  `label_json` json COMMENT '该块上的标注汇总 [{label_code,value,...}]',
  `text_edit_status` smallint NOT NULL COMMENT '是否经人工修改：0-否 1-是',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  KEY idx_dwd_opr_content_block_di_0 (tenant_code, asset_no),
  KEY idx_dwd_opr_content_block_di_1 (tenant_code, category_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='入库内容块明细（核对后最终文本，分区日期=入库日期）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS dwd.dwd_opr_knowledge_chunk_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `chunk_no` varchar(64) NOT NULL COMMENT '切片编号',
  `kb_code` varchar(64) NOT NULL COMMENT '知识库编码',
  `asset_no` varchar(64) NOT NULL COMMENT '来源资产编号',
  `block_no_list` longtext NOT NULL COMMENT '组成该切片的内容块编号，逗号分隔',
  `chunk_text` longtext NOT NULL COMMENT '切片文本（含标题路径前缀）',
  `chunk_token_qty` int NOT NULL COMMENT '切片 token 数',
  `page_no` int COMMENT '起始页码',
  `start_ms_qty` bigint COMMENT '起始时间（毫秒，音视频切片）',
  `embedding_vector` json COMMENT '向量（PostgreSQL 使用 pgvector；MySQL 以 JSON 文本存储，不参与检索）',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  KEY idx_dwd_opr_knowledge_chunk_di_0 (tenant_code, kb_code),
  KEY idx_dwd_opr_knowledge_chunk_di_1 (tenant_code, asset_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识切片明细（含向量，供智能体检索，分区日期=入库日期）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS dwd.dwd_opr_content_asset_zip (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `asset_no` varchar(64) NOT NULL COMMENT '资产编号',
  `modality_type` varchar(16) NOT NULL COMMENT '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频',
  `category_code` varchar(64) COMMENT '分类编码',
  `stage_code` smallint NOT NULL COMMENT '流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用',
  `stage_status` smallint NOT NULL COMMENT '阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消',
  `start_date` date NOT NULL COMMENT '拉链开始日期（含）',
  `end_date` date NOT NULL COMMENT '拉链结束日期（不含），当前有效记录为 9999-12-31',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  KEY idx_dwd_opr_content_asset_zip_0 (tenant_code, asset_no, end_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容资产拉链表（记录分类、区域、阶段的历史变更）';

CREATE TABLE IF NOT EXISTS dwd.dwd_cus_user_zip (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `user_code` varchar(64) NOT NULL COMMENT '用户编码',
  `user_type` smallint NOT NULL COMMENT '用户类型：0-平台超级管理员 1-租户管理员 2-普通用户 3-API 账号',
  `user_status` smallint NOT NULL COMMENT '用户状态：0-正常 1-锁定 2-停用',
  `role_code_list` varchar(1024) COMMENT '当时拥有的角色编码，逗号分隔',
  `start_date` date NOT NULL COMMENT '拉链开始日期（含）',
  `end_date` date NOT NULL COMMENT '拉链结束日期（不含），当前有效记录为 9999-12-31',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  KEY idx_dwd_cus_user_zip_0 (tenant_code, user_code, end_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户拉链表（记录用户状态、类型、归属区域的历史变更）';

-- ======================================================================
-- dws
-- ======================================================================

CREATE TABLE IF NOT EXISTS dws.dws_opr_content_process_di (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `modality_type` varchar(16) NOT NULL COMMENT '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频',
  `upload_qty` int NOT NULL DEFAULT 0 COMMENT '上传数',
  `parse_success_qty` int NOT NULL DEFAULT 0 COMMENT '解析成功数',
  `parse_fail_qty` int NOT NULL DEFAULT 0 COMMENT '解析失败数（重试用尽）',
  `retry_qty` int NOT NULL DEFAULT 0 COMMENT '重试次数合计',
  `review_qty` int NOT NULL DEFAULT 0 COMMENT '核对完成数',
  `review_edit_block_qty` int NOT NULL DEFAULT 0 COMMENT '核对中被修改的内容块数',
  `label_qty` int NOT NULL DEFAULT 0 COMMENT '标注条数',
  `store_qty` int NOT NULL DEFAULT 0 COMMENT '入库数',
  `apply_qty` int NOT NULL DEFAULT 0 COMMENT '应用数',
  `avg_parse_ms_qty` bigint COMMENT '平均解析耗时（毫秒）',
  `review_pass_rate` decimal(6,4) COMMENT '核对一次通过率（未修改块占比）',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  UNIQUE KEY uk_dws_opr_content_process_di_0 (tenant_code, region_code, modality_type, travel_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容处理日汇总（租户×区域×模态×日）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS dws.dws_opr_content_stage_hrs (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `stage_hour_code` char(2) NOT NULL COMMENT '小时，00~23',
  `stage_code` smallint NOT NULL COMMENT '流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用',
  `enter_qty` int NOT NULL DEFAULT 0 COMMENT '进入该阶段的资产数',
  `finish_qty` int NOT NULL DEFAULT 0 COMMENT '完成该阶段的资产数',
  `fail_qty` int NOT NULL DEFAULT 0 COMMENT '失败数',
  `backlog_qty` int NOT NULL DEFAULT 0 COMMENT '小时末积压数',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  UNIQUE KEY uk_dws_opr_content_stage_hrs_0 (tenant_code, region_code, stage_hour_code, stage_code, travel_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='流水线阶段小时吞吐（租户×阶段×小时，用于队列积压监控）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS dws.dws_fin_tenant_usage_mo (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `stat_month_code` char(6) NOT NULL COMMENT '统计月份 YYYYMM',
  `storage_mb_qty` bigint NOT NULL DEFAULT 0 COMMENT '月末存储量（MB）',
  `parse_page_qty` bigint NOT NULL DEFAULT 0 COMMENT '解析页数',
  `media_minute_qty` bigint NOT NULL DEFAULT 0 COMMENT '音视频处理分钟数',
  `input_token_qty` bigint NOT NULL DEFAULT 0 COMMENT '智能体输入 token',
  `output_token_qty` bigint NOT NULL DEFAULT 0 COMMENT '智能体输出 token',
  `cost_amt` decimal(18,6) COMMENT '估算模型费用（美元）',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  UNIQUE KEY uk_dws_fin_tenant_usage_mo_0 (tenant_code, stat_month_code, travel_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='租户月度用量汇总（存储、解析页数、音视频时长、智能体 token，用于计费）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

-- ======================================================================
-- ads
-- ======================================================================

CREATE TABLE IF NOT EXISTS ads.ads_opr_content_dashboard_df (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `total_asset_qty` bigint NOT NULL DEFAULT 0 COMMENT '累计资产数',
  `today_upload_qty` int NOT NULL DEFAULT 0 COMMENT '当日上传数',
  `pending_review_qty` int NOT NULL DEFAULT 0 COMMENT '待核对数',
  `pending_label_qty` int NOT NULL DEFAULT 0 COMMENT '待标注数',
  `failed_qty` int NOT NULL DEFAULT 0 COMMENT '失败待处理数',
  `stored_qty` bigint NOT NULL DEFAULT 0 COMMENT '累计入库数',
  `kb_chunk_qty` bigint NOT NULL DEFAULT 0 COMMENT '累计知识切片数',
  `parse_success_rate` decimal(6,4) COMMENT '当日解析成功率',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  UNIQUE KEY uk_ads_opr_content_dashboard_df_0 (tenant_code, region_code, travel_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容中心看板指标（天全量快照：截至当日的累计与当日指标）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

CREATE TABLE IF NOT EXISTS ads.ads_opr_content_quality_mo (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `stat_month_code` char(6) NOT NULL COMMENT '统计月份 YYYYMM',
  `modality_type` varchar(16) NOT NULL COMMENT '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频',
  `category_code` varchar(64) NOT NULL COMMENT '分类编码',
  `asset_qty` int NOT NULL DEFAULT 0 COMMENT '资产数',
  `review_pass_rate` decimal(6,4) COMMENT '核对一次通过率',
  `avg_review_minute_qty` decimal(10,2) COMMENT '平均人工核对时长（分钟）',
  `auto_classify_accept_rate` decimal(6,4) COMMENT '模型分类建议采纳率',
  `travel_date` char(8) NOT NULL COMMENT '分区日期，格式YYYYMMDD',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id, travel_date),
  UNIQUE KEY uk_ads_opr_content_quality_mo_0 (tenant_code, region_code, stat_month_code, modality_type, category_code, travel_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容质量月报（模态×分类维度的解析质量与人工工作量）'
PARTITION BY RANGE COLUMNS(travel_date) (PARTITION p_max VALUES LESS THAN (MAXVALUE));

-- ======================================================================
-- dim
-- ======================================================================

CREATE TABLE IF NOT EXISTS dim.dim_cus_region_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）',
  `region_name` varchar(128) NOT NULL COMMENT '区域名称',
  `parent_region_code` varchar(32) NOT NULL DEFAULT '' COMMENT '上级区域编码，根节点为空串',
  `region_path` varchar(512) NOT NULL COMMENT '区域路径，形如 /CN/CN-31/CN-31-01/，用于前缀匹配下级',
  `region_level` smallint NOT NULL COMMENT '区域层级：1-国家 2-省/州 3-城市 4-区县 5-景区/门店',
  `region_type` varchar(16) NOT NULL DEFAULT 'admin' COMMENT '区域类型：admin-行政区 scenic-景区 store-门店 virtual-虚拟分组',
  `region_sort_qty` int COMMENT '同级排序号，越小越靠前',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_dim_cus_region_f_0 (tenant_code, region_code),
  KEY idx_dim_cus_region_f_0 (tenant_code, region_path)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='区域维度表（树形，区域权限的授权单元）';

CREATE TABLE IF NOT EXISTS dim.dim_opr_content_modality_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `modality_type` varchar(16) NOT NULL COMMENT '模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频',
  `modality_name` varchar(64) NOT NULL COMMENT '模态中文名称',
  `file_ext_list` varchar(256) NOT NULL COMMENT '允许的文件扩展名，逗号分隔，如 pdf 或 doc,docx',
  `parser_type` varchar(32) NOT NULL COMMENT '解析通道：mineru-MinerU文档解析 markdown-Markdown直读 media-音视频(ffmpeg+ASR+抽帧经MinerU)',
  `max_file_size_mb_qty` int NOT NULL COMMENT '单文件大小上限（MB）',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_dim_opr_content_modality_f_0 (tenant_code, modality_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容模态字典（七种模态及其解析通道）';

CREATE TABLE IF NOT EXISTS dim.dim_opr_content_stage_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `stage_code` smallint NOT NULL COMMENT '流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用',
  `stage_name` varchar(64) NOT NULL COMMENT '阶段中文名称',
  `stage_type` varchar(16) NOT NULL COMMENT '执行方式：auto-系统自动 manual-人工 hybrid-系统建议+人工确认',
  `next_stage_code` smallint NOT NULL COMMENT '默认下一阶段编码，0 表示流程结束',
  `skip_allowed_status` smallint NOT NULL DEFAULT 0 COMMENT '是否允许跳过：0-不允许 1-允许',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_dim_opr_content_stage_f_0 (tenant_code, stage_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='流水线阶段字典';

CREATE TABLE IF NOT EXISTS dim.dim_opr_content_category_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `category_code` varchar(64) NOT NULL COMMENT '分类编码',
  `category_name` varchar(128) NOT NULL COMMENT '分类名称',
  `parent_category_code` varchar(64) NOT NULL DEFAULT '' COMMENT '上级分类编码，根为空串',
  `category_path` varchar(512) NOT NULL COMMENT '分类路径，形如 /contract/supplier/',
  `target_table_name` varchar(128) COMMENT '入库回填目标表（库名.表名），为空表示只进知识库，如 ods.ods_fin_invoice_extract_di',
  `extract_prompt` longtext COMMENT '回填字段抽取提示词（交给智能体辅助模型）',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_dim_opr_content_category_f_0 (tenant_code, category_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='内容分类体系（树形，租户可自定义）';

CREATE TABLE IF NOT EXISTS dim.dim_opr_label_schema_f (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tenant_code` varchar(32) NOT NULL COMMENT '租户编码（SaaS 租户隔离）',
  `label_code` varchar(64) NOT NULL COMMENT '标签编码',
  `label_name` varchar(128) NOT NULL COMMENT '标签名称',
  `label_type` smallint NOT NULL COMMENT '标注形态：1-文本实体 2-图像区域框 3-音视频时间片 4-整篇标签 5-表格单元格',
  `label_color` varchar(16) COMMENT '前端展示颜色，如 #409EFF',
  `value_schema` json COMMENT '标签取值约束（JSON Schema），为空表示自由文本',
  `record_status` smallint NOT NULL DEFAULT 0 COMMENT '记录状态：0-正常 1-已删除（逻辑删除）',
  `create_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '创建人用户编码，系统任务为 system',
  `create_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  `update_user_code` varchar(64) NOT NULL DEFAULT 'system' COMMENT '最后更新人用户编码',
  `update_time` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '最后更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_dim_opr_label_schema_f_0 (tenant_code, label_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='标注体系定义（标签及其可用的标注形态）';
