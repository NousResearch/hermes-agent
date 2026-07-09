import os


def resolve_path_within_directory(
    base_dir: str,
    unsafe_path: str,
    *,
    require_file: bool = True,
) -> str:
    # 用户传入的路径可能是文件名、相对路径、绝对路径，也可能夹带 `../`。
    # 这里统一解析成真实路径，并用 commonpath 判断它是否仍在允许目录内。
    # 这样比简单判断字符串前缀可靠，可以覆盖符号链接、重复分隔符、相对路径
    # 等场景，适用于上传目录、素材目录、任务产物目录这类白名单目录。
    if not unsafe_path:
        raise ValueError("empty path is not allowed")

    base_dir_real = os.path.realpath(base_dir)
    candidate_path = unsafe_path
    if not os.path.isabs(candidate_path):
        candidate_path = os.path.join(base_dir_real, candidate_path)

    resolved_path = os.path.realpath(candidate_path)
    try:
        common_path = os.path.commonpath([base_dir_real, resolved_path])
    except ValueError as exc:
        # Windows 下不同盘符会触发 ValueError，这类路径一定不属于允许目录。
        raise ValueError("path is outside the allowed directory") from exc

    if common_path != base_dir_real:
        raise ValueError("path is outside the allowed directory")

    if require_file and not os.path.isfile(resolved_path):
        raise ValueError("file does not exist")

    return resolved_path
