# shellcheck shell=sh
# Installed as /etc/profile.d/hermes.sh. Debian's /etc/profile resets PATH for
# every login shell (`bash -l`, `docker exec <c> bash -l`, ssh, the terminal
# tool's background `bash -lic`), dropping the image's ENV PATH dirs, so bare
# `hermes` exits 127 there. Put them back in the ENV PATH order; dirs a login
# PATH already lists keep their position.
for __hermes_p in /opt/data/.local/bin /opt/hermes/.venv/bin /opt/hermes/bin; do
    case ":$PATH:" in
        *":$__hermes_p:"*) ;;
        *) PATH="$__hermes_p:$PATH" ;;
    esac
done
unset __hermes_p
export PATH
