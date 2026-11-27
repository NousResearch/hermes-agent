"""Config-backed scratch placement and fail-closed mount admission.

This is admission, not recovery: a vanished persisted workspace needs operator
recovery rather than creation of an empty replacement.
"""
import os
import secrets
from pathlib import Path


class WorkspaceUnavailable(ValueError):
    """Workspace admission refused without charging a worker failure."""


def configured_root():
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly().get("kanban", {})
    raw = config.get("workspaces_root")
    required = config.get("workspaces_root_require_mount", False)
    if not isinstance(required, bool):
        raise WorkspaceUnavailable("workspaces_root_invalid: require_mount must be boolean")
    if raw in (None, ""):
        if required:
            raise WorkspaceUnavailable("workspaces_root_invalid: mount guard requires a root")
        return None, False
    if not isinstance(raw, str) or not Path(raw).expanduser().is_absolute():
        raise WorkspaceUnavailable("workspaces_root_invalid: root must be absolute")
    return Path(raw).expanduser(), required


def validate_mount(root, *, expected_mount=None):
    # Permit a pre-provisioned directory directly below a mount, as well as
    # the mount itself. Never walk upwards to / and call the SSD a valid mount.
    if not root.is_dir() or root.is_symlink():
        raise WorkspaceUnavailable(f"workspaces_root_unmounted: {root}")
    filesystem_root = Path(root.anchor)
    if root == filesystem_root:
        raise WorkspaceUnavailable("workspaces_root_invalid: filesystem root is not scratch")
    root_is_mount = os.path.ismount(root)
    parent_is_mount = root.parent != filesystem_root and os.path.ismount(root.parent)
    if expected_mount is not None:
        expected_mount = Path(expected_mount)
        if expected_mount not in (root, root.parent) or not os.path.ismount(expected_mount):
            raise WorkspaceUnavailable(f"workspaces_root_unmounted: {root}")
        mount_path = expected_mount
    elif root_is_mount:
        mount_path = root
    elif parent_is_mount:
        mount_path = root.parent
    else:
        raise WorkspaceUnavailable(f"workspaces_root_unmounted: {root}")
    try:
        resolved = root.resolve()
    except (OSError, RuntimeError) as exc:
        raise WorkspaceUnavailable(f"workspaces_root_invalid: cannot resolve: {root}") from exc
    if resolved != root.absolute():
        raise WorkspaceUnavailable(f"workspaces_root_invalid: symlink ancestor: {root}")

    # Reject a read-only or wedged mount before a task is claimed. Permission
    # bits are not authoritative on ACL/noowners mounts, so exercise the actual
    # write path through a no-follow directory descriptor.
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    probe_name = f".hermes-write-probe-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        root_fd = os.open(root, flags)
        try:
            probe_fd = os.open(
                probe_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=root_fd,
            )
            os.close(probe_fd)
            os.unlink(probe_name, dir_fd=root_fd)
        finally:
            os.close(root_fd)
    except OSError as exc:
        raise WorkspaceUnavailable(f"workspaces_root_unwritable: {root}") from exc
    return mount_path


def validate_persisted(path):
    try:
        exists = path.is_dir()
    except (OSError, RuntimeError) as exc:
        raise WorkspaceUnavailable(f"stranded_by_mount_loss: {path}") from exc
    if not exists:
        raise WorkspaceUnavailable(f"stranded_by_mount_loss: {path}")


def validate_target(root, path):
    """Reject existing symlink/cross-device components without creating them."""
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise WorkspaceUnavailable("workspaces_root_invalid: target outside root") from exc
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        fd = os.open(root, flags)
    except OSError as exc:
        raise WorkspaceUnavailable(f"workspaces_root_unmounted: {root}") from exc
    try:
        root_dev = os.fstat(fd).st_dev
        for part in parts:
            if part in (".", ".."):
                raise WorkspaceUnavailable("workspaces_root_invalid: traversal")
            try:
                child = os.open(part, flags, dir_fd=fd)
            except FileNotFoundError:
                return
            except OSError as exc:
                raise WorkspaceUnavailable(
                    f"workspaces_root_invalid: unsafe component: {part}"
                ) from exc
            os.close(fd)
            fd = child
            if os.fstat(fd).st_dev != root_dev:
                raise WorkspaceUnavailable("workspaces_root_invalid: nested filesystem")
    finally:
        os.close(fd)


def create_scratch(root, path, *, expected_mount):
    """Create through a pinned directory FD using the admitted mount anchor."""
    validate_mount(root, expected_mount=expected_mount)
    validate_target(root, path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(root, flags)
    try:
        validate_mount(root, expected_mount=expected_mount)
        opened = os.fstat(fd)
        current = root.stat()
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise WorkspaceUnavailable(f"workspaces_root_unmounted: changed during admission: {root}")
        for part in path.relative_to(root).parts:
            if part in (".", ".."):
                raise WorkspaceUnavailable("workspaces_root_invalid: traversal")
            try:
                os.mkdir(part, mode=0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
            if os.fstat(fd).st_dev != opened.st_dev:
                raise WorkspaceUnavailable("workspaces_root_invalid: nested filesystem")
        validate_mount(root, expected_mount=expected_mount)
    finally:
        os.close(fd)
