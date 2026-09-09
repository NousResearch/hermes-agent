"""CLI ownership of durable conversation roots, carried from e828efe6ec."""
import atexit
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def _interactive_cli_conversation_worktree_applies() -> bool:
    """Return whether this process owns a user-started CLI conversation root."""
    source = os.environ.get("HERMES_SESSION_SOURCE", "cli").strip().lower() or "cli"
    return source == "cli" and not os.environ.get("HERMES_KANBAN_TASK", "").strip()


def _build_cli_conversation_worktree_manager(config, db):
    """Build the shared manager only for policy-owned interactive CLI roots."""
    from agent.conversation_worktree import (
        ConversationWorktreeError,
        ConversationWorktreeManager,
    )
    from agent.conversation_worktree_policy import resolve_conversation_worktree_policy

    policy = resolve_conversation_worktree_policy(config)
    if not policy.enabled or not _interactive_cli_conversation_worktree_applies():
        return None
    if db is None:
        raise ConversationWorktreeError("state.db is unavailable", phase="state")
    return ConversationWorktreeManager(policy, db)


def _should_use_legacy_worktree(*, worktree: bool, shorthand: bool, config) -> bool:
    """Keep manual ``-w`` ownership separate from managed conversation roots."""
    requested = bool(worktree or shorthand or config.get("worktree", False))
    if not requested:
        return False

    from agent.conversation_worktree_policy import resolve_conversation_worktree_policy

    policy = resolve_conversation_worktree_policy(config)
    managed_root = policy.enabled and _interactive_cli_conversation_worktree_applies()
    return not managed_root


def _cli_conversation_worktree_prompt_fragment(binding) -> str:
    """Match the desktop's certified-worktree instruction for CLI roots."""
    return (
        "This interactive conversation is isolated in a certified Git worktree. "
        f"Use {binding.path} as its workspace (branch {binding.branch or 'unknown'}, "
        f"base {binding.base_commit or 'unknown'}). "
        "Do not switch to the stable source checkout."
    )


class CLIConversationWorktreeMixin:
    def _conversation_worktree_root(self, session_id=None):
        """Compression inherits a workspace; explicit forks start their own root."""
        current = session_id or self.session_id
        seen = set()
        while current not in seen:
            seen.add(current)
            row = self._session_db.get_session(current)
            if not row or self._session_db.is_explicit_fork_child(current):
                return current
            parent = row.get("parent_session_id")
            if not parent:
                return current
            current = parent
        from agent.conversation_worktree import ConversationWorktreeError

        raise ConversationWorktreeError("cyclic CLI conversation lineage", phase="state")

    def _apply_conversation_worktree_binding(self, binding, *, before_commit=None) -> None:
        """Retarget CLI-owned tools and prompt context to a certified binding."""
        managed_path = str(binding.path)
        prior_cwd = os.getcwd() if before_commit is not None else None
        try:
            os.chdir(managed_path)
        except OSError as exc:
            from agent.conversation_worktree import ConversationWorktreeError

            raise ConversationWorktreeError(
                f"could not enter managed conversation worktree {managed_path}: {exc}",
                phase="cwd",
            ) from exc

        if before_commit is not None:
            try:
                before_commit()
            except Exception:
                os.chdir(prior_cwd)
                raise

        prior_note = getattr(self, "_conversation_worktree_prompt_note", "")
        if prior_note:
            rendered_prior = f"\n\n[System note: {prior_note}]"
            self.system_prompt = (self.system_prompt or "").replace(rendered_prior, "")

        note = _cli_conversation_worktree_prompt_fragment(binding)
        self._conversation_worktree_binding = binding
        self._conversation_worktree_prompt_note = note
        self.working_directory = managed_path
        os.environ["TERMINAL_CWD"] = self.working_directory
        self.system_prompt = (self.system_prompt or "") + f"\n\n[System note: {note}]"
        if self.agent is not None:
            self.agent.ephemeral_system_prompt = self.system_prompt

    @staticmethod
    def _acquire_conversation_root_lease(binding, *, surface: str):
        from agent.conversation_worktree import acquire_conversation_root_lease

        return acquire_conversation_root_lease(
            root_session_id=str(binding.root_session_id),
            worktree_path=Path(binding.path),
            repo_common_dir=Path(binding.repo_common_dir),
            surface=surface,
        )

    def _initialize_conversation_worktree(self, config, resume, manage_conversation_worktree):
        self._conversation_worktree_manager = (
            _build_cli_conversation_worktree_manager(config, self._session_db)
            if manage_conversation_worktree
            else None
        )
        self._conversation_worktree_binding = None
        self._conversation_worktree_prompt_note = ""
        self._conversation_root_lease = None
        if self._conversation_worktree_manager is not None:
            if resume:
                try:
                    root_session_id = self._conversation_worktree_root()
                except Exception as exc:
                    from agent.conversation_worktree import ConversationWorktreeError

                    raise ConversationWorktreeError(
                        "could not resolve the durable CLI conversation root",
                        phase="state",
                    ) from exc
                binding = self._conversation_worktree_manager.resolve_existing_session(
                    root_session_id
                )
                if binding is None:
                    from agent.conversation_worktree import ConversationWorktreeError

                    raise ConversationWorktreeError(
                        f"no ready conversation worktree for CLI root {root_session_id}",
                        phase="recovery",
                    )
            else:
                # A new CLI session is only a draft until its first prompt.  Do not
                # create a retained manager-owned worktree for a process that exits
                # before the user ever sends a message.
                binding = None
            if binding is not None:
                self._apply_conversation_worktree_binding(binding)
                self._conversation_root_lease = self._acquire_conversation_root_lease(
                    binding, surface="cli"
                )
            atexit.register(self._release_active_session)

    def _ensure_conversation_worktree_binding(self):
        """Bind a new CLI root when its first prompt makes the session durable."""
        manager = getattr(self, "_conversation_worktree_manager", None)
        if manager is None or getattr(self, "_conversation_worktree_binding", None) is not None:
            return getattr(self, "_conversation_worktree_binding", None)

        from agent.conversation_worktree import ConversationWorktreeError

        binding = manager.bind_new_root_session(
            self.session_id, conversation_kind="interactive"
        )
        if binding is None:
            raise ConversationWorktreeError(
                f"conversation worktree policy did not bind CLI root {self.session_id}",
                phase="create",
            )
        self._apply_conversation_worktree_binding(binding)
        self._conversation_root_lease = self._acquire_conversation_root_lease(
            binding, surface="cli"
        )
        return binding

    def _restore_managed_conversation_cwd(self, *, session_id=None):
        managed_binding = getattr(self, "_conversation_worktree_binding", None)
        if managed_binding is not None:
            # Persisted cwd from an older session row is subordinate to the
            # durable manager binding. Resolve from the current session id so
            # an in-process /resume targets the selected conversation's root,
            # not the binding from the session that was just left.
            try:
                root_session_id = self._conversation_worktree_root(session_id)
                managed_binding = (
                    self._conversation_worktree_manager.resolve_existing_session(
                        root_session_id
                    )
                )
            except Exception as exc:
                from agent.conversation_worktree import ConversationWorktreeError

                raise ConversationWorktreeError(
                    "could not resolve the resumed CLI conversation worktree",
                    phase="state",
                ) from exc
            if managed_binding is None:
                from agent.conversation_worktree import ConversationWorktreeError

                raise ConversationWorktreeError(
                    f"no ready conversation worktree for CLI root {root_session_id}",
                    phase="recovery",
                )
            next_root_lease = self._acquire_conversation_root_lease(
                managed_binding, surface="cli"
            )
            try:
                self._apply_conversation_worktree_binding(managed_binding)
            except Exception:
                next_root_lease.release()
                raise
            prior_root_lease = getattr(self, "_conversation_root_lease", None)
            self._conversation_root_lease = next_root_lease
            if prior_root_lease is not None:
                try:
                    prior_root_lease.release()
                except Exception:
                    logger.debug("Failed to release prior root lease", exc_info=True)
            return True

        return False

    def _prepare_conversation_root(self, new_session_id, *, before_commit=None):
        from cli import _cprint
        new_worktree_binding = None

        # Claim, certify, and enter the new root before finalizing, flushing,
        # or resetting the current conversation. A failed create or cwd
        # transition leaves every old-session identity and in-memory state
        # untouched and usable.
        conversation_worktree_manager = getattr(
            self, "_conversation_worktree_manager", None
        )
        if conversation_worktree_manager is not None:
            try:
                new_worktree_binding = (
                    conversation_worktree_manager.bind_new_root_session(
                        new_session_id, conversation_kind="interactive"
                    )
                )
                if new_worktree_binding is None:
                    raise RuntimeError("manager returned no conversation worktree binding")
                new_root_lease = self._acquire_conversation_root_lease(
                    new_worktree_binding, surface="cli"
                )
                try:
                    self._apply_conversation_worktree_binding(
                        new_worktree_binding, before_commit=before_commit
                    )
                except Exception:
                    new_root_lease.release()
                    raise
                prior_root_lease = getattr(self, "_conversation_root_lease", None)
                self._conversation_root_lease = new_root_lease
                if prior_root_lease is not None:
                    try:
                        prior_root_lease.release()
                    except Exception:
                        logger.debug("Failed to release prior root lease", exc_info=True)
            except Exception as exc:
                _cprint(
                    f"  Cannot start new session {new_session_id}: "
                    f"conversation worktree setup failed: {exc}"
                )
                return False

        return True
