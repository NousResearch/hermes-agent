# shellcheck shell=sh
# Installed as /etc/profile.d/hermes.sh. Debian's /etc/profile resets PATH for
# every login shell (`bash -l`, `docker exec <c> bash -l`, ssh, the terminal
# tool's background `bash -lic`), dropping the image's ENV PATH dirs, so bare
# `hermes` exits 127 there. Put them back in the ENV PATH order; dirs a login
# PATH already lists keep their position.
for __hermes_p in /opt/hermes/.venv/bin /opt/hermes/bin; do
    case ":$PATH:" in
        *":$__hermes_p:"*) ;;
        *) PATH="$__hermes_p:$PATH" ;;
    esac
done
# The hermes user's ~/.local/bin (profile wrappers, pip --user, uv tool) lives
# on the hermes-writable data volume: append it after the system dirs, and only
# for non-root shells, so a root login shell never resolves a command there.
if [ "$(/usr/bin/id -u)" != 0 ]; then
    case ":$PATH:" in
        *":/opt/data/.local/bin:"*) ;;
        *) PATH="$PATH:/opt/data/.local/bin" ;;
    esac
fi
unset __hermes_p
export PATH
