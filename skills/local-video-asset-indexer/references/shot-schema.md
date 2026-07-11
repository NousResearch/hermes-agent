# 镜头索引字段

每个镜头必须可追溯到原视频和时间范围。

## 程序生成字段

- `asset_id`：由源文件内容 SHA-256 派生。
- `clip_id`：镜头记录 ID。
- `source_path`：原视频绝对路径。
- `start_seconds` / `end_seconds` / `duration_seconds`：镜头范围。
- `keyframe_path`：关键帧路径。
- `file_path`：按需裁切后的实体文件；未裁切时为空。
- `materialized`：是否已经生成实体镜头文件。

这些字段不能由模型猜测。

## AI 生成字段

- `description`：具体、可检索的一句话画面描述。
- `semantic_json.content`：主体、场景、动作、工序、可见品牌元素。
- `semantic_json.cinematography`：景别、机位、可确认的运镜和画面特点。
- `semantic_json.creative`：情绪、商业用途、建议文案。
- `quality_score`：0-1 综合质量排序值。
- `confidence`：0-1 画面理解置信度。
- `tags`：固定标签与自由标签。

静态关键帧无法可靠确认真实运镜或声音时，对应字段必须留空。

## 状态

- `ready`：可参与检索与匹配。
- `low_confidence`：可检索但默认降权。
- `unusable`：全黑、严重模糊、损坏等，默认排除。
- `semantic_failed`：技术切镜成功但视觉分析失败，可重试。
