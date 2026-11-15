"""PDR 全量表结构定义（单一事实来源）。

DDL 文件 deploy/sql/{postgres,mysql}/*.sql 由 `python -m pdr_server.db.render` 从这里生成，
不要手改 SQL。命名规范检查见 naming.py。

库（schema）按数据分层划分：src / ods / dwd / dws / ads / dim，表名带库名引用，如
`dwd.dwd_opr_content_block_di`。主题域前缀：权限与租户归 customer(cus)，平台运营类元数据
（配置、内容流水线、知识库、智能体用量）归 operation(opr)，入库回填的业务抽取表归各业务域
（fin / sup / prd），租户计费归 finance(fin)。
"""

from __future__ import annotations

from .spec import PARTITION_COLUMN, Col, TableSpec, c_audit, c_id, c_region, c_tenant, c_travel_date

TABLES: list[TableSpec] = []


def table(
    schema: str,
    name: str,
    comment: str,
    cols: list[Col],
    *,
    region: bool = True,
    indexes: list[list[str]] | None = None,
    uniques: list[list[str]] | None = None,
) -> TableSpec:
    """统一拼装：id + tenant_code [+ region_code] + 业务字段 [+ travel_date] + 审计字段。"""
    head = [c_id(), c_tenant(), *([c_region()] if region else [])]
    spec = TableSpec(schema, name, comment, head + cols + c_audit(), indexes=indexes or [], uniques=uniques or [])
    if spec.partitioned:
        spec.columns.insert(len(head) + len(cols), c_travel_date())
    TABLES.append(spec)
    return spec


def C(name: str, type_: str, comment: str, nullable: bool = True, default: str | None = None) -> Col:
    return Col(name, type_, comment, nullable=nullable, default=default)


def NN(name: str, type_: str, comment: str, default: str | None = None) -> Col:
    return Col(name, type_, comment, nullable=False, default=default)


# 通用取值说明（字段注释里逐值写明含义）
MODALITY = "模态类型：image-图片 pdf-PDF excel-Excel word-Word markdown-Markdown video-视频 audio-音频"
STAGE = "流水线阶段：10-上传 20-解析 30-人工核对 40-标注 50-分类 60-入库 70-应用"
STAGE_STATUS = (
    "阶段状态：0-待处理 1-处理中 2-成功 3-失败 4-待重试 5-已驳回 6-已跳过 9-已取消"
)
BLOCK_TYPE = (
    "内容块类型：title-标题 text-正文 list-列表 table-表格 image-图片 equation-公式 code-代码 "
    "header-页眉 footer-页脚 sheet-工作表 frame-视频关键帧 speech-语音片段"
)

# ===========================================================================
# DIM 维度层
# ===========================================================================

table("dim", "dim_cus_region_f", "区域维度表（树形，区域权限的授权单元）", [
    NN("region_name", "varchar(128)", "区域名称"),
    NN("parent_region_code", "varchar(32)", "上级区域编码，根节点为空串", "''"),
    NN("region_path", "varchar(512)", "区域路径，形如 /CN/CN-31/CN-31-01/，用于前缀匹配下级"),
    NN("region_level", "smallint", "区域层级：1-国家 2-省/州 3-城市 4-区县 5-景区/门店"),
    NN("region_type", "varchar(16)", "区域类型：admin-行政区 scenic-景区 store-门店 virtual-虚拟分组", "'admin'"),
    C("region_sort_qty", "int", "同级排序号，越小越靠前", "0"),
], region=True, uniques=[["tenant_code", "region_code"]], indexes=[["tenant_code", "region_path"]])

table("dim", "dim_opr_content_modality_f", "内容模态字典（七种模态及其解析通道）", [
    NN("modality_type", "varchar(16)", MODALITY),
    NN("modality_name", "varchar(64)", "模态中文名称"),
    NN("file_ext_list", "varchar(256)", "允许的文件扩展名，逗号分隔，如 pdf 或 doc,docx"),
    NN("parser_type", "varchar(32)", "解析通道：mineru-MinerU文档解析 markdown-Markdown直读 media-音视频(ffmpeg+ASR+抽帧经MinerU)"),
    NN("max_file_size_mb_qty", "int", "单文件大小上限（MB）"),
], region=False, uniques=[["tenant_code", "modality_type"]])

table("dim", "dim_opr_content_stage_f", "流水线阶段字典", [
    NN("stage_code", "smallint", STAGE),
    NN("stage_name", "varchar(64)", "阶段中文名称"),
    NN("stage_type", "varchar(16)", "执行方式：auto-系统自动 manual-人工 hybrid-系统建议+人工确认"),
    NN("next_stage_code", "smallint", "默认下一阶段编码，0 表示流程结束"),
    NN("skip_allowed_status", "smallint", "是否允许跳过：0-不允许 1-允许", "0"),
], region=False, uniques=[["tenant_code", "stage_code"]])

table("dim", "dim_opr_content_category_f", "内容分类体系（树形，租户可自定义）", [
    NN("category_code", "varchar(64)", "分类编码"),
    NN("category_name", "varchar(128)", "分类名称"),
    NN("parent_category_code", "varchar(64)", "上级分类编码，根为空串", "''"),
    NN("category_path", "varchar(512)", "分类路径，形如 /contract/supplier/"),
    C("target_table_name", "varchar(128)", "入库回填目标表（库名.表名），为空表示只进知识库，如 ods.ods_fin_invoice_extract_di"),
    C("extract_prompt", "text", "回填字段抽取提示词（交给智能体辅助模型）"),
], region=False, uniques=[["tenant_code", "category_code"]])

table("dim", "dim_opr_label_schema_f", "标注体系定义（标签及其可用的标注形态）", [
    NN("label_code", "varchar(64)", "标签编码"),
    NN("label_name", "varchar(128)", "标签名称"),
    NN("label_type", "smallint", "标注形态：1-文本实体 2-图像区域框 3-音视频时间片 4-整篇标签 5-表格单元格"),
    C("label_color", "varchar(16)", "前端展示颜色，如 #409EFF"),
    C("value_schema", "json", "标签取值约束（JSON Schema），为空表示自由文本"),
], region=False, uniques=[["tenant_code", "label_code"]])

# ===========================================================================
# SRC 业务源层 —— 租户与权限（customer 域）
# ===========================================================================

table("src", "src_cus_tenant_f", "租户表（SaaS 租户主数据）", [
    NN("tenant_name", "varchar(128)", "租户名称"),
    NN("tenant_type", "smallint", "租户类型：0-平台运营方 1-企业租户 2-试用租户"),
    NN("tenant_status", "smallint", "租户状态：0-正常 1-停用 2-已过期"),
    NN("agent_profile_name", "varchar(64)", "对应 hermes 智能体 profile 名称，形如 t_<tenant_code>"),
    C("expire_date", "date", "服务到期日期"),
    C("max_user_qty", "int", "最大用户数配额"),
    C("max_storage_gb_qty", "int", "存储配额（GB）"),
    C("contact_name", "varchar(64)", "联系人姓名"),
    C("contact_phone_no", "varchar(32)", "联系人电话号码"),
], region=False, uniques=[["tenant_code"]])

table("src", "src_cus_user_f", "用户表", [
    NN("user_code", "varchar(64)", "用户编码（租户内唯一，登录名）"),
    NN("user_name", "varchar(64)", "用户姓名"),
    NN("password_hash", "varchar(256)", "密码哈希（argon2id）"),
    C("email_addr", "varchar(128)", "电子邮箱地址"),
    C("mobile_no", "varchar(32)", "手机号码"),
    NN("user_type", "smallint", "用户类型：0-平台超级管理员 1-租户管理员 2-普通用户 3-API 账号"),
    NN("user_status", "smallint", "用户状态：0-正常 1-锁定 2-停用"),
    C("last_login_time", "timestamp", "最近登录时间"),
], uniques=[["tenant_code", "user_code"]])

table("src", "src_cus_role_f", "角色表", [
    NN("role_code", "varchar(64)", "角色编码"),
    NN("role_name", "varchar(64)", "角色名称"),
    NN("role_type", "smallint", "角色类型：0-系统内置（不可删） 1-租户自定义"),
    C("role_desc", "varchar(512)", "角色说明"),
], region=False, uniques=[["tenant_code", "role_code"]])

table("src", "src_cus_permission_f", "权限点表（菜单/按钮/接口，全平台共享，tenant_code=platform）", [
    NN("permission_code", "varchar(128)", "权限点编码，形如 content:review:submit"),
    NN("permission_name", "varchar(128)", "权限点名称"),
    NN("permission_type", "smallint", "权限点类型：1-菜单 2-按钮 3-接口"),
    NN("parent_permission_code", "varchar(128)", "上级权限点编码", "''"),
    C("api_method_name", "varchar(16)", "接口 HTTP 方法（类型为接口时）：GET/POST/PUT/PATCH/DELETE"),
    C("api_path_name", "varchar(256)", "接口路径模板（类型为接口时），如 /api/content/assets/{id}"),
    C("route_path_name", "varchar(256)", "前端路由路径（类型为菜单时）"),
], region=False, uniques=[["tenant_code", "permission_code"]])

table("src", "src_cus_user_role_f", "用户-角色关系表", [
    NN("user_code", "varchar(64)", "用户编码"),
    NN("role_code", "varchar(64)", "角色编码"),
], region=False, uniques=[["tenant_code", "user_code", "role_code"]])

table("src", "src_cus_role_permission_f", "角色-权限点关系表", [
    NN("role_code", "varchar(64)", "角色编码"),
    NN("permission_code", "varchar(128)", "权限点编码"),
], region=False, uniques=[["tenant_code", "role_code", "permission_code"]])

table("src", "src_cus_user_region_f", "用户归属区域表（用户可归属多个区域，作为“本区域”取值来源）", [
    NN("user_code", "varchar(64)", "用户编码"),
    NN("primary_status", "smallint", "是否主区域：0-否 1-是", "0"),
], uniques=[["tenant_code", "user_code", "region_code"]])

table("src", "src_cus_role_data_scope_f", "角色数据范围表（行级权限：按资源类型配置可见范围）", [
    NN("role_code", "varchar(64)", "角色编码"),
    NN("resource_type", "varchar(64)", "资源类型：content_asset-内容资产 knowledge-知识库 business_extract-业务回填数据 report-报表 *-全部"),
    NN("scope_type", "smallint", "范围类型：1-本租户全部 2-本区域及下级 3-仅本区域 4-自定义区域 5-仅本人创建"),
    NN("action_type", "varchar(16)", "适用操作：read-读 write-写 *-读写", "'*'"),
], region=False, uniques=[["tenant_code", "role_code", "resource_type", "action_type"]])

table("src", "src_cus_role_region_f", "角色自定义区域表（数据范围为 4-自定义区域 时生效）", [
    NN("role_code", "varchar(64)", "角色编码"),
    NN("resource_type", "varchar(64)", "资源类型：content_asset-内容资产 knowledge-知识库 business_extract-业务回填数据 report-报表 *-全部"),
    NN("include_child_status", "smallint", "是否包含下级区域：0-否 1-是", "1"),
], uniques=[["tenant_code", "role_code", "resource_type", "region_code"]])

table("src", "src_cus_role_column_perm_f", "角色列级权限表（未配置的列默认可读写）", [
    NN("role_code", "varchar(64)", "角色编码"),
    NN("table_name", "varchar(128)", "受控表（库名.表名），如 ods.ods_fin_invoice_extract_di"),
    NN("column_name", "varchar(128)", "受控字段名"),
    NN("perm_type", "smallint", "列权限：0-不可见 1-脱敏可见 2-只读 3-可读写"),
    C("mask_rule_type", "varchar(32)", "脱敏规则：mobile-手机号 id_card-证件号 amount-金额区间 name-姓名 full-全遮蔽"),
], region=False, uniques=[["tenant_code", "role_code", "table_name", "column_name"]])

table("src", "src_cus_api_token_f", "API 访问令牌表（供 API 账号与外部系统调用）", [
    NN("user_code", "varchar(64)", "所属用户编码"),
    NN("token_name", "varchar(64)", "令牌名称"),
    NN("token_hash", "varchar(128)", "令牌 SHA-256 哈希（明文只在创建时展示一次）"),
    NN("token_prefix_code", "varchar(16)", "令牌前缀（用于界面识别）"),
    C("expire_time", "timestamp", "过期时间，为空表示不过期"),
    NN("token_status", "smallint", "令牌状态：0-有效 1-已吊销"),
], region=False, uniques=[["token_hash"]])

table("src", "src_cus_operate_log_di", "操作审计日志（天增量）", [
    NN("user_code", "varchar(64)", "操作人用户编码"),
    NN("operate_type", "varchar(64)", "操作类型，开放取值，与权限点编码一致，如 config:update / content:review:submit / role:grant"),
    C("resource_type", "varchar(64)", "操作对象资源类型：content_asset-内容资产 knowledge-知识库 config-配置 user-用户 role-角色 tenant-租户 business_extract-业务回填数据"),
    C("resource_no", "varchar(128)", "操作对象编号"),
    C("request_path_name", "varchar(256)", "请求接口路径"),
    C("client_ip_code", "varchar(64)", "客户端 IP"),
    NN("operate_status", "smallint", "操作结果：0-成功 1-失败 2-拒绝（无权限）"),
    C("detail_json", "json", "操作详情（变更前后值等，敏感字段已脱敏）"),
    NN("operate_time", "timestamp", "操作时间"),
], indexes=[["tenant_code", "user_code", "operate_time"]])

# ===========================================================================
# SRC —— 配置中心（operation 域）
# ===========================================================================

table("src", "src_opr_config_item_f", "配置项字典（hermes 智能体与 PDR 系统配置项的中文元数据）", [
    NN("config_key", "varchar(256)", "配置键（点分路径），如 hermes.agent.max_turns / pdr.pipeline.retry.max_attempt_qty"),
    NN("config_domain_type", "varchar(16)", "配置归属：pdr-PDR系统配置 hermes-智能体配置"),
    NN("group_code", "varchar(64)", "页面分组编码（取配置键第一段，如 agent / compression）"),
    NN("group_name", "varchar(64)", "页面分组中文名"),
    NN("config_name", "varchar(128)", "配置项中文名称"),
    C("config_desc", "text", "配置项中文说明"),
    NN("value_type", "varchar(16)", "取值类型：bool-布尔 int-整数 float-小数 str-字符串 enum-枚举 list-列表 map-键值对 secret-密钥引用"),
    C("default_value", "json", "默认值（JSON 编码）"),
    C("value_range", "json", "取值范围：{min,max} 或 {enum:[{value,label}]} 或 {pattern}"),
    NN("reload_type", "smallint", "生效方式：0-立即热加载 1-下一会话生效（保护提示缓存） 2-需重启服务"),
    NN("scope_type", "smallint", "可配置层级：0-仅平台 1-平台与租户均可"),
    C("sort_qty", "int", "组内排序号", "0"),
], region=False, uniques=[["tenant_code", "config_key"]])

table("src", "src_opr_config_value_f", "租户配置覆盖值（平台级值以 config.yaml 为准，本表只存租户覆盖）", [
    NN("config_key", "varchar(256)", "配置键，关联 src.src_opr_config_item_f.config_key"),
    C("config_value", "json", "覆盖值（JSON 编码）"),
    NN("value_version_qty", "int", "版本号，每次保存 +1，用于乐观锁", "1"),
], region=False, uniques=[["tenant_code", "config_key"]])

table("src", "src_opr_config_change_log_di", "配置变更记录（平台级与租户级修改均记录，天增量）", [
    NN("config_key", "varchar(256)", "配置键"),
    NN("change_scope_type", "smallint", "变更层级：0-平台（写入 config.yaml） 1-租户（写入覆盖表）"),
    C("before_value", "json", "变更前值（密钥类只记录是否设置）"),
    C("after_value", "json", "变更后值（密钥类只记录是否设置）"),
    NN("apply_status", "smallint", "生效状态：0-已生效 1-待下一会话 2-待重启 3-生效失败"),
    C("config_file_sha256_code", "char(64)", "写入后 config.yaml 的 SHA-256，便于核对页面与文件一致"),
    NN("change_time", "timestamp", "变更时间"),
], region=False, indexes=[["tenant_code", "config_key", "change_time"]])

# ===========================================================================
# SRC —— 多模态内容流水线（operation 域）
# ===========================================================================

table("src", "src_opr_content_batch_f", "内容上传批次表", [
    NN("batch_no", "varchar(64)", "批次编号"),
    NN("batch_name", "varchar(128)", "批次名称"),
    NN("file_qty", "int", "批次文件数", "0"),
    NN("batch_status", "smallint", "批次状态：0-上传中 1-已提交 2-处理中 3-全部完成 4-部分失败"),
    C("default_category_code", "varchar(64)", "批次默认分类编码"),
], uniques=[["tenant_code", "batch_no"]])

table("src", "src_opr_content_asset_f", "内容资产主表（一个上传文件一行，记录当前阶段与状态）", [
    NN("asset_no", "varchar(64)", "资产编号"),
    C("batch_no", "varchar(64)", "所属批次编号"),
    NN("asset_name", "varchar(256)", "资产名称（默认取原文件名）"),
    NN("modality_type", "varchar(16)", MODALITY),
    NN("file_ext_code", "varchar(16)", "文件扩展名，如 pdf / docx / mp4"),
    NN("file_size_qty", "bigint", "文件大小（字节）"),
    NN("file_sha256_code", "char(64)", "文件 SHA-256（租户内去重）"),
    NN("storage_uri_name", "varchar(512)", "原件存储地址，如 file:///data/... 或 s3://bucket/key"),
    C("page_qty", "int", "页数/工作表数（文档类）"),
    C("duration_ms_qty", "bigint", "时长（毫秒，音视频类）"),
    NN("stage_code", "smallint", "当前" + STAGE),
    NN("stage_status", "smallint", "当前" + STAGE_STATUS),
    C("category_code", "varchar(64)", "确认后的分类编码"),
    C("assignee_user_code", "varchar(64)", "当前人工阶段处理人用户编码"),
    C("parse_backend_name", "varchar(64)", "解析所用后端，如 mineru:standard / markdown / media"),
    C("business_date", "date", "业务日期（如发票日期、合同签署日期），可由抽取回填"),
], uniques=[["tenant_code", "asset_no"]],
    indexes=[["tenant_code", "region_code", "stage_code", "stage_status"], ["tenant_code", "file_sha256_code"]])

table("src", "src_opr_content_object_f", "内容派生文件表（转换 PDF、页面图、裁切图、关键帧、音轨、字幕、导出包等）", [
    NN("asset_no", "varchar(64)", "所属资产编号"),
    NN("object_type", "varchar(32)", "派生类型：page_image-页面图 crop_image-版面裁切图 frame-视频关键帧 audio_track-音轨 subtitle_srt-SRT字幕 subtitle_vtt-VTT字幕 mineru_raw-MinerU原始产物包 export-导出包"),
    NN("storage_uri_name", "varchar(512)", "存储地址"),
    C("mime_type", "varchar(128)", "MIME 类型，开放取值，遵循 IANA 媒体类型，如 application/pdf"),
    C("file_size_qty", "bigint", "文件大小（字节）"),
    C("page_no", "int", "关联页码（从 1 开始）"),
    C("time_ms_qty", "bigint", "关联时间点（毫秒，关键帧用）"),
], indexes=[["tenant_code", "asset_no", "object_type"]])

table("src", "src_opr_content_media_f", "音视频元数据表（ffprobe 抽取）", [
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("duration_ms_qty", "bigint", "时长（毫秒）"),
    C("container_name", "varchar(32)", "封装格式，如 mp4 / mkv / mp3 / wav"),
    C("video_codec_name", "varchar(32)", "视频编码，如 h264"),
    C("audio_codec_name", "varchar(32)", "音频编码，如 aac"),
    C("width_qty", "int", "画面宽度（像素）"),
    C("height_qty", "int", "画面高度（像素）"),
    C("frame_per_sec_qty", "decimal(8,3)", "帧率（帧/秒）"),
    C("sample_rate_qty", "int", "音频采样率（Hz）"),
    C("channel_qty", "smallint", "音频声道数"),
    C("speech_lang_code", "varchar(16)", "识别出的语种，如 zh / en"),
    C("frame_qty", "int", "抽取关键帧数"),
    C("segment_qty", "int", "转写语音片段数"),
], uniques=[["tenant_code", "asset_no"]])

table("src", "src_opr_content_stage_task_f", "流水线阶段任务表（每个资产每个阶段一次执行一行，含重试）", [
    NN("task_no", "varchar(64)", "任务编号"),
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("stage_code", "smallint", STAGE),
    NN("task_status", "smallint", "任务" + STAGE_STATUS),
    NN("attempt_qty", "int", "已执行次数", "0"),
    NN("max_attempt_qty", "int", "最大执行次数（来自 config.yaml pipeline.retry）", "3"),
    C("next_retry_time", "timestamp", "下次重试时间（指数退避）"),
    C("queue_message_code", "varchar(64)", "Redis Stream 消息 ID"),
    C("worker_name", "varchar(128)", "执行该任务的 worker 名称"),
    C("operator_user_code", "varchar(64)", "人工阶段的操作人用户编码；自动阶段为 system"),
    C("start_time", "timestamp", "开始时间"),
    C("end_time", "timestamp", "结束时间"),
    C("error_code", "varchar(64)", "失败错误码，如 MINERU_TIMEOUT / ASR_FAILED"),
    C("error_msg", "text", "失败详情"),
    C("task_param_json", "json", "任务参数（解析档位、OCR 语言等）"),
], uniques=[["tenant_code", "task_no"]], indexes=[["tenant_code", "asset_no", "stage_code"], ["task_status", "next_retry_time"]])

table("src", "src_opr_content_stage_log_di", "资产状态流转日志（每次状态变化一行，天增量）", [
    NN("asset_no", "varchar(64)", "资产编号"),
    C("task_no", "varchar(64)", "关联任务编号"),
    NN("from_stage_code", "smallint", "变更前" + STAGE + "，0-新建"),
    NN("from_stage_status", "smallint", "变更前" + STAGE_STATUS),
    NN("to_stage_code", "smallint", "变更后" + STAGE),
    NN("to_stage_status", "smallint", "变更后" + STAGE_STATUS),
    NN("operator_user_code", "varchar(64)", "操作人用户编码，系统自动为 system"),
    C("change_reason", "varchar(512)", "变更原因（驳回意见、失败原因、跳过说明）"),
    NN("change_time", "timestamp", "变更时间"),
], indexes=[["tenant_code", "asset_no", "change_time"]])

table("src", "src_opr_content_block_f", "内容块工作表（解析结果的结构化内容，供人工核对、标注；入库后同步到 DWD）", [
    NN("block_no", "varchar(64)", "内容块编号"),
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("block_type", "varchar(16)", BLOCK_TYPE),
    NN("block_order_qty", "int", "阅读顺序号（资产内从 1 开始）"),
    C("parent_block_no", "varchar(64)", "父块编号（表格标题/图片说明挂在表格/图片上）"),
    C("page_no", "int", "页码/工作表序号（从 1 开始）"),
    C("bbox_json", "json", "版面坐标 [x0,y0,x1,y1]（页面归一化到 0~1000）"),
    C("title_level_qty", "smallint", "标题层级（block_type=title 时 1~6）"),
    C("start_ms_qty", "bigint", "开始时间（毫秒，speech/frame 用）"),
    C("end_ms_qty", "bigint", "结束时间（毫秒，speech 用）"),
    C("speaker_code", "varchar(32)", "说话人标识（语音片段，若开启说话人分离）"),
    C("origin_text", "text", "解析原始文本（不可改，便于对比）"),
    C("review_text", "text", "人工核对后文本，为空表示与原文一致"),
    C("table_html", "text", "表格 HTML（block_type=table/sheet）"),
    C("table_json", "json", "表格结构化数据（二维数组，含合并单元格信息）"),
    C("image_object_id", "bigint", "关联图片派生文件 id（src.src_opr_content_object_f.id）"),
    C("confidence_rate", "decimal(5,4)", "解析/识别置信度 0~1"),
    NN("review_status", "smallint", "核对状态：0-未核对 1-已确认无误 2-已修改 3-标记删除", "0"),
], uniques=[["tenant_code", "block_no"]], indexes=[["tenant_code", "asset_no", "block_order_qty"]])

table("src", "src_opr_media_align_f", "音视频对齐表（关键帧与语音片段按时间重叠对齐，用于图文字幕联合检索）", [
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("frame_block_no", "varchar(64)", "关键帧内容块编号（block_type=frame）"),
    NN("speech_block_no", "varchar(64)", "语音片段内容块编号（block_type=speech）"),
    NN("overlap_ms_qty", "bigint", "两者时间重叠时长（毫秒）"),
    NN("align_type", "smallint", "对齐方式：1-时间重叠 2-最近邻（帧落在片段间隙时） 3-人工调整"),
], uniques=[["tenant_code", "frame_block_no", "speech_block_no"]])

table("src", "src_opr_content_label_f", "标注记录表", [
    NN("label_no", "varchar(64)", "标注编号"),
    NN("asset_no", "varchar(64)", "资产编号"),
    C("block_no", "varchar(64)", "内容块编号，整篇标签为空"),
    NN("label_code", "varchar(64)", "标签编码（dim.dim_opr_label_schema_f）"),
    NN("label_type", "smallint", "标注形态：1-文本实体 2-图像区域框 3-音视频时间片 4-整篇标签 5-表格单元格"),
    C("label_value", "text", "标注值（实体文本或用户填写值）"),
    C("char_start_qty", "int", "文本实体起始字符偏移"),
    C("char_end_qty", "int", "文本实体结束字符偏移（不含）"),
    C("bbox_json", "json", "图像区域框 [x0,y0,x1,y1]"),
    C("start_ms_qty", "bigint", "时间片开始（毫秒）"),
    C("end_ms_qty", "bigint", "时间片结束（毫秒）"),
    C("cell_json", "json", "表格单元格坐标 {row,col}"),
    NN("label_source_type", "smallint", "标注来源：0-人工 1-模型预标注 2-模型预标注经人工确认"),
], uniques=[["tenant_code", "label_no"]], indexes=[["tenant_code", "asset_no"]])

table("src", "src_opr_content_category_rel_f", "资产分类关系表（含模型建议与人工确认）", [
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("category_code", "varchar(64)", "分类编码"),
    NN("classify_source_type", "smallint", "分类来源：0-人工 1-规则 2-模型建议"),
    C("confidence_rate", "decimal(5,4)", "模型建议置信度 0~1"),
    NN("confirm_status", "smallint", "确认状态：0-待确认 1-已采纳 2-已否决"),
], uniques=[["tenant_code", "asset_no", "category_code", "classify_source_type"]])

table("src", "src_opr_knowledge_base_f", "知识库表（入库内容的应用载体，挂载到租户智能体）", [
    NN("kb_code", "varchar(64)", "知识库编码"),
    NN("kb_name", "varchar(128)", "知识库名称"),
    C("kb_desc", "varchar(512)", "知识库说明（提供给智能体的工具描述）"),
    NN("embedding_model_name", "varchar(128)", "向量模型名称"),
    NN("embedding_dim_qty", "int", "向量维度"),
    NN("kb_status", "smallint", "知识库状态：0-启用 1-停用"),
], uniques=[["tenant_code", "kb_code"]])

table("src", "src_opr_content_application_f", "内容应用记录表（入库内容被投放到哪些应用）", [
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("apply_type", "smallint", "应用类型：1-知识库检索 2-业务表回填 3-导出下载 4-智能体会话附件"),
    NN("target_code", "varchar(256)", "应用目标：知识库编码 / 回填表（库名.表名） / 导出格式"),
    NN("apply_status", "smallint", "应用状态：0-待执行 1-执行中 2-成功 3-失败 4-已撤回"),
    C("apply_result_json", "json", "执行结果（切片数、回填行数、导出地址等）"),
    C("apply_time", "timestamp", "执行完成时间"),
], uniques=[["tenant_code", "asset_no", "apply_type", "target_code"]])

# ===========================================================================
# ODS 原始数据层（解析器原样输出，只追加不修改）
# ===========================================================================

table("ods", "ods_opr_mineru_parse_result_di", "MinerU 原始解析结果（每次解析一行，分区日期=解析日期）", [
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("task_no", "varchar(64)", "解析任务编号"),
    NN("parser_type", "varchar(32)", "解析通道：mineru-MinerU文档解析 markdown-Markdown直读 media-音视频管线"),
    C("mineru_version_name", "varchar(32)", "MinerU 版本号"),
    C("mineru_tier_name", "varchar(16)", "MinerU 档位：flash / basic / standard / advanced"),
    C("mineru_task_code", "varchar(64)", "MinerU 服务侧任务 ID"),
    C("middle_json_uri_name", "varchar(512)", "middle_json.json 存储地址"),
    C("markdown_uri_name", "varchar(512)", "markdown.md 存储地址"),
    C("structured_json_uri_name", "varchar(512)", "structured_content.json 存储地址"),
    C("block_qty", "int", "解析出的内容块数"),
    C("parse_duration_ms_qty", "bigint", "解析耗时（毫秒）"),
    NN("parse_status", "smallint", "解析结果：0-成功 1-失败 2-部分成功"),
], indexes=[["tenant_code", "asset_no"]])

table("ods", "ods_opr_media_transcript_di", "语音转写原始片段（ASR 原样输出，分区日期=转写日期）", [
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("task_no", "varchar(64)", "转写任务编号"),
    NN("segment_order_qty", "int", "片段顺序号"),
    NN("start_ms_qty", "bigint", "开始时间（毫秒）"),
    NN("end_ms_qty", "bigint", "结束时间（毫秒）"),
    C("speaker_code", "varchar(32)", "说话人标识"),
    NN("segment_text", "text", "转写文本"),
    C("word_json", "json", "词级时间戳 [{word,start_ms,end_ms,prob}]"),
    C("avg_logprob_score", "decimal(8,4)", "平均对数概率（ASR 置信参考，越接近 0 越可信）"),
    C("asr_model_name", "varchar(64)", "ASR 模型名称，如 faster-whisper-large-v3"),
], indexes=[["tenant_code", "asset_no"]])

table("ods", "ods_opr_media_frame_di", "视频抽帧原始记录（分区日期=抽帧日期）", [
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("task_no", "varchar(64)", "抽帧任务编号"),
    NN("frame_order_qty", "int", "帧顺序号"),
    NN("time_ms_qty", "bigint", "帧时间点（毫秒）"),
    NN("extract_type", "smallint", "抽帧方式：1-场景切换 2-固定间隔 3-人工补帧"),
    C("scene_change_score", "decimal(6,4)", "场景切换分值 0~1"),
    NN("storage_uri_name", "varchar(512)", "帧图片存储地址"),
    C("ocr_text", "text", "帧图片经 MinerU 识别的文字"),
    C("caption_text", "text", "视觉模型生成的画面描述（可选）"),
], indexes=[["tenant_code", "asset_no"]])

table("ods", "ods_opr_agent_session_di", "智能体会话用量原始记录（从各租户 hermes profile 同步，分区日期=会话日期）", [
    NN("session_code", "varchar(128)", "hermes 会话 ID"),
    NN("user_code", "varchar(64)", "发起用户编码"),
    C("model_name", "varchar(128)", "使用的模型"),
    C("provider_name", "varchar(64)", "模型提供商"),
    NN("input_token_qty", "bigint", "输入 token 数", "0"),
    NN("output_token_qty", "bigint", "输出 token 数", "0"),
    NN("cache_read_token_qty", "bigint", "缓存命中 token 数", "0"),
    C("cost_amt", "decimal(18,6)", "估算费用（美元）"),
    C("session_start_time", "timestamp", "会话开始时间"),
], uniques=[["tenant_code", "session_code"]])

# 业务回填抽取表（分类 target_table_name 指向这里；字段由抽取提示词约定）
table("ods", "ods_fin_invoice_extract_di", "发票抽取原始数据（发票类内容入库回填，分区日期=开票日期）", [
    NN("asset_no", "varchar(64)", "来源资产编号"),
    C("invoice_no", "varchar(64)", "发票号码"),
    C("invoice_code", "varchar(64)", "发票代码"),
    C("invoice_type", "varchar(32)", "发票类型：vat_special-增值税专票 vat_normal-增值税普票 e_invoice-全电发票 itinerary-航空行程单 other-其他"),
    C("invoice_date", "date", "开票日期"),
    C("seller_name", "varchar(256)", "销售方名称"),
    C("buyer_name", "varchar(256)", "购买方名称"),
    C("invoice_amt", "decimal(18,2)", "价税合计金额"),
    C("tax_amt", "decimal(18,2)", "税额"),
    C("tax_rate", "decimal(6,4)", "税率"),
    NN("verify_status", "smallint", "人工核验状态：0-待核验 1-已核验 2-有误", "0"),
], indexes=[["tenant_code", "invoice_no"]])

table("ods", "ods_sup_contract_extract_di", "供应商合同抽取原始数据（分区日期=签署日期）", [
    NN("asset_no", "varchar(64)", "来源资产编号"),
    C("contract_no", "varchar(64)", "合同编号"),
    C("sup_name", "varchar(256)", "供应商名称"),
    C("sup_type", "varchar(32)", "供应商类型：hotel-酒店 airline-航司 ground-地接 attraction-景点 other-其他"),
    C("sign_date", "date", "签署日期"),
    C("start_date", "date", "合作开始日期"),
    C("end_date", "date", "合作结束日期"),
    C("settlement_type", "varchar(32)", "结算方式：prepay-预付 monthly-月结 per_order-单结"),
    C("net_price_amt", "decimal(18,2)", "结算价金额（合同主报价）"),
    C("commission_rate", "decimal(6,4)", "佣金比例"),
    NN("verify_status", "smallint", "人工核验状态：0-待核验 1-已核验 2-有误", "0"),
], indexes=[["tenant_code", "contract_no"]])

table("ods", "ods_prd_product_extract_di", "产品资料抽取原始数据（线路/酒店/门票宣传资料，分区日期=资料生效日期）", [
    NN("asset_no", "varchar(64)", "来源资产编号"),
    C("prd_name", "varchar(256)", "产品名称"),
    C("prd_type", "varchar(32)", "产品类型：flight-机票 hotel-酒店 tour-跟团游 combo-组合产品 attraction-景点门票"),
    C("attraction_name", "varchar(256)", "主要景点名称"),
    C("tour_day_qty", "int", "行程天数"),
    C("adult_price_amt", "decimal(18,2)", "成人价"),
    C("child_price_amt", "decimal(18,2)", "儿童价"),
    C("valid_start_date", "date", "有效期开始日期"),
    C("valid_end_date", "date", "有效期结束日期"),
    NN("verify_status", "smallint", "人工核验状态：0-待核验 1-已核验 2-有误", "0"),
])

# ===========================================================================
# DWD 明细层（入库：核对/标注/分类确认后的规范化内容）
# ===========================================================================

table("dwd", "dwd_opr_content_block_di", "入库内容块明细（核对后最终文本，分区日期=入库日期）", [
    NN("block_no", "varchar(64)", "内容块编号"),
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("modality_type", "varchar(16)", MODALITY),
    NN("category_code", "varchar(64)", "资产分类编码"),
    NN("block_type", "varchar(16)", BLOCK_TYPE),
    NN("block_order_qty", "int", "阅读顺序号"),
    C("page_no", "int", "页码"),
    C("start_ms_qty", "bigint", "开始时间（毫秒）"),
    C("end_ms_qty", "bigint", "结束时间（毫秒）"),
    C("final_text", "text", "最终文本（核对文本优先，否则原始文本）"),
    C("table_json", "json", "表格结构化数据"),
    C("label_json", "json", "该块上的标注汇总 [{label_code,value,...}]"),
    NN("text_edit_status", "smallint", "是否经人工修改：0-否 1-是"),
], indexes=[["tenant_code", "asset_no"], ["tenant_code", "category_code"]])

table("dwd", "dwd_opr_knowledge_chunk_di", "知识切片明细（含向量，供智能体检索，分区日期=入库日期）", [
    NN("chunk_no", "varchar(64)", "切片编号"),
    NN("kb_code", "varchar(64)", "知识库编码"),
    NN("asset_no", "varchar(64)", "来源资产编号"),
    NN("block_no_list", "text", "组成该切片的内容块编号，逗号分隔"),
    NN("chunk_text", "text", "切片文本（含标题路径前缀）"),
    NN("chunk_token_qty", "int", "切片 token 数"),
    C("page_no", "int", "起始页码"),
    C("start_ms_qty", "bigint", "起始时间（毫秒，音视频切片）"),
    C("embedding_vector", "vector", "向量（PostgreSQL 使用 pgvector；MySQL 以 JSON 文本存储，不参与检索）"),
], indexes=[["tenant_code", "kb_code"], ["tenant_code", "asset_no"]])

table("dwd", "dwd_opr_content_asset_zip", "内容资产拉链表（记录分类、区域、阶段的历史变更）", [
    NN("asset_no", "varchar(64)", "资产编号"),
    NN("modality_type", "varchar(16)", MODALITY),
    C("category_code", "varchar(64)", "分类编码"),
    NN("stage_code", "smallint", STAGE),
    NN("stage_status", "smallint", STAGE_STATUS),
    NN("start_date", "date", "拉链开始日期（含）"),
    NN("end_date", "date", "拉链结束日期（不含），当前有效记录为 9999-12-31"),
], indexes=[["tenant_code", "asset_no", "end_date"]])

table("dwd", "dwd_cus_user_zip", "用户拉链表（记录用户状态、类型、归属区域的历史变更）", [
    NN("user_code", "varchar(64)", "用户编码"),
    NN("user_type", "smallint", "用户类型：0-平台超级管理员 1-租户管理员 2-普通用户 3-API 账号"),
    NN("user_status", "smallint", "用户状态：0-正常 1-锁定 2-停用"),
    C("role_code_list", "varchar(1024)", "当时拥有的角色编码，逗号分隔"),
    NN("start_date", "date", "拉链开始日期（含）"),
    NN("end_date", "date", "拉链结束日期（不含），当前有效记录为 9999-12-31"),
], indexes=[["tenant_code", "user_code", "end_date"]])

# ===========================================================================
# DWS 汇总层
# ===========================================================================

table("dws", "dws_opr_content_process_di", "内容处理日汇总（租户×区域×模态×日）", [
    NN("modality_type", "varchar(16)", MODALITY),
    NN("upload_qty", "int", "上传数", "0"),
    NN("parse_success_qty", "int", "解析成功数", "0"),
    NN("parse_fail_qty", "int", "解析失败数（重试用尽）", "0"),
    NN("retry_qty", "int", "重试次数合计", "0"),
    NN("review_qty", "int", "核对完成数", "0"),
    NN("review_edit_block_qty", "int", "核对中被修改的内容块数", "0"),
    NN("label_qty", "int", "标注条数", "0"),
    NN("store_qty", "int", "入库数", "0"),
    NN("apply_qty", "int", "应用数", "0"),
    C("avg_parse_ms_qty", "bigint", "平均解析耗时（毫秒）"),
    C("review_pass_rate", "decimal(6,4)", "核对一次通过率（未修改块占比）"),
], uniques=[["tenant_code", "region_code", "modality_type"]])

table("dws", "dws_opr_content_stage_hrs", "流水线阶段小时吞吐（租户×阶段×小时，用于队列积压监控）", [
    NN("stage_hour_code", "char(2)", "小时，00~23"),
    NN("stage_code", "smallint", STAGE),
    NN("enter_qty", "int", "进入该阶段的资产数", "0"),
    NN("finish_qty", "int", "完成该阶段的资产数", "0"),
    NN("fail_qty", "int", "失败数", "0"),
    NN("backlog_qty", "int", "小时末积压数", "0"),
], uniques=[["tenant_code", "region_code", "stage_hour_code", "stage_code"]])

table("dws", "dws_fin_tenant_usage_mo", "租户月度用量汇总（存储、解析页数、音视频时长、智能体 token，用于计费）", [
    NN("stat_month_code", "char(6)", "统计月份 YYYYMM"),
    NN("storage_mb_qty", "bigint", "月末存储量（MB）", "0"),
    NN("parse_page_qty", "bigint", "解析页数", "0"),
    NN("media_minute_qty", "bigint", "音视频处理分钟数", "0"),
    NN("input_token_qty", "bigint", "智能体输入 token", "0"),
    NN("output_token_qty", "bigint", "智能体输出 token", "0"),
    C("cost_amt", "decimal(18,6)", "估算模型费用（美元）"),
], region=False, uniques=[["tenant_code", "stat_month_code"]])

# ===========================================================================
# ADS 应用层（看板 / 报表；可同步到 MySQL 报表库）
# ===========================================================================

table("ads", "ads_opr_content_dashboard_df", "内容中心看板指标（天全量快照：截至当日的累计与当日指标）", [
    NN("total_asset_qty", "bigint", "累计资产数", "0"),
    NN("today_upload_qty", "int", "当日上传数", "0"),
    NN("pending_review_qty", "int", "待核对数", "0"),
    NN("pending_label_qty", "int", "待标注数", "0"),
    NN("failed_qty", "int", "失败待处理数", "0"),
    NN("stored_qty", "bigint", "累计入库数", "0"),
    NN("kb_chunk_qty", "bigint", "累计知识切片数", "0"),
    C("parse_success_rate", "decimal(6,4)", "当日解析成功率"),
], uniques=[["tenant_code", "region_code"]])

table("ads", "ads_opr_content_quality_mo", "内容质量月报（模态×分类维度的解析质量与人工工作量）", [
    NN("stat_month_code", "char(6)", "统计月份 YYYYMM"),
    NN("modality_type", "varchar(16)", MODALITY),
    NN("category_code", "varchar(64)", "分类编码"),
    NN("asset_qty", "int", "资产数", "0"),
    C("review_pass_rate", "decimal(6,4)", "核对一次通过率"),
    C("avg_review_minute_qty", "decimal(10,2)", "平均人工核对时长（分钟）"),
    C("auto_classify_accept_rate", "decimal(6,4)", "模型分类建议采纳率"),
], uniques=[["tenant_code", "region_code", "stat_month_code", "modality_type", "category_code"]])


def by_full_name() -> dict[str, TableSpec]:
    return {t.full_name: t for t in TABLES}


assert all(PARTITION_COLUMN in t.column_names() for t in TABLES if t.partitioned)
