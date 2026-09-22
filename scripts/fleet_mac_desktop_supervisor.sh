#!/bin/zsh

set -u

if [[ $# -ne 5 ]]; then
    print -u2 "usage: $0 DESKTOP_EXECUTABLE HERMES_ROOT PYTHON_EXECUTABLE COORDINATOR_SOURCE_ROOT POLL_SECONDS"
    exit 2
fi

desktop_executable="$1"
hermes_root="$2"
python_executable="$3"
source_root="$4"
poll_seconds="$5"
fleet_root="$hermes_root/fleet"
marker="$fleet_root/desktop-live"
token_file="$hermes_root/.env"
coordinator_pid_file="$fleet_root/coordinator.pid"
runner_pid_file="$fleet_root/mac-runner.pid"
coordinator_log="$fleet_root/coordinator.log"
runner_log="$fleet_root/mac-runner.log"

mkdir -p "$fleet_root"

desktop_is_live() {
    /bin/ps -axo pid=,command= | /usr/bin/awk -v expected="$desktop_executable" '
        {
            pid = $1
            $1 = ""
            sub(/^[[:space:]]+/, "")
            if ($0 == expected) {
                print pid
                exit
            }
        }
    '
}

pid_is_live() {
    local pid="$1"
    [[ "$pid" == <-> ]] && /bin/kill -0 "$pid" 2>/dev/null
}

pid_from_file() {
    local path="$1"
    [[ -f "$path" ]] && /bin/cat "$path" || true
}

stop_pid_file() {
    local path="$1"
    local pid="$(pid_from_file "$path")"
    if pid_is_live "$pid"; then
        /bin/kill "$pid" 2>/dev/null || true
    fi
    /bin/rm -f "$path"
}

read_token() {
    /usr/bin/awk -F= '$1 == "HERMES_FLEET_TOKEN" { sub(/^[^=]*=/, ""); print; exit }' "$token_file"
}

start_coordinator() {
    local token="$(read_token)"
    [[ -n "$token" ]] || return 1
    export HERMES_FLEET_TOKEN="$token"
    HERMES_FLEET_TOKEN="$token" /usr/bin/nohup "$python_executable" \
        "$source_root/scripts/fleet_coordinator.py" \
        --db "$fleet_root/fleet.db" --host 0.0.0.0 --port 8799 \
        >> "$coordinator_log" 2>&1 &
    print $! >| "$coordinator_pid_file"
}

start_runner() {
    local token="$(read_token)"
    [[ -n "$token" ]] || return 1
    export HERMES_FLEET_TOKEN="$token"
    typeset -a profile_args
    typeset -a profile_names
    typeset -a profile_model_args
    typeset -a profile_provider_args
    typeset -A seen_profiles
    profile_args=()
    profile_names=()
    profile_model_args=()
    profile_provider_args=()
    for profile_name in default coding-expert task-orchestrator; do
        if [[ -z "${seen_profiles[$profile_name]-}" ]]; then
            profile_names+=("$profile_name")
            seen_profiles[$profile_name]=1
        fi
    done
    for profile_dir in "$hermes_root/profiles"/*(/N); do
        profile_name="${profile_dir:t}"
        if [[ -z "${seen_profiles[$profile_name]-}" ]]; then
            profile_names+=("$profile_name")
            seen_profiles[$profile_name]=1
        fi
    done
    for profile_name in "${profile_names[@]}"; do
        profile_args+=(--profile "$profile_name")
        profile_config="$hermes_root/config.yaml"
        [[ "$profile_name" != "default" ]] && profile_config="$hermes_root/profiles/$profile_name/config.yaml"
        model_name="$(/usr/bin/awk '
            /^model:[[:space:]]*$/ { in_model=1; next }
            /^[^[:space:]]/ { in_model=0 }
            in_model && /^[[:space:]]+default:[[:space:]]*/ {
                sub(/^[[:space:]]+default:[[:space:]]*/, "")
                print
                exit
            }
        ' "$profile_config" 2>/dev/null | /usr/bin/sed -E 's/^['"'"']|['"'"']$//g')"
        provider_name="$(/usr/bin/awk '
            /^model:[[:space:]]*$/ { in_model=1; next }
            /^[^[:space:]]/ { in_model=0 }
            in_model && /^[[:space:]]+provider:[[:space:]]*/ {
                sub(/^[[:space:]]+provider:[[:space:]]*/, "")
                print
                exit
            }
        ' "$profile_config" 2>/dev/null | /usr/bin/sed -E 's/^['"'"']|['"'"']$//g')"
        if [[ -n "$model_name" && -n "$provider_name" ]]; then
            [[ -n "$model_name" ]] && profile_model_args+=(--profile-model "$profile_name=$model_name")
            [[ -n "$provider_name" ]] && profile_provider_args+=(--profile-provider "$profile_name=$provider_name")
        fi
    done
    HERMES_FLEET_TOKEN="$token" /usr/bin/nohup "$python_executable" \
        "$source_root/scripts/fleet_runner.py" \
        --node-id mac --coordinator http://127.0.0.1:8799 \
        --hermes-executable /Users/mikedemott/.local/bin/hermes \
        "${profile_args[@]}" \
        --project "Hermes Agent" --project LunaBot \
        "${profile_model_args[@]}" "${profile_provider_args[@]}" \
        --liveness-file "$marker" --interval 2 \
        >> "$runner_log" 2>&1 &
    print $! >| "$runner_pid_file"
}

stop_stack() {
    stop_pid_file "$runner_pid_file"
    stop_pid_file "$coordinator_pid_file"
    /bin/rm -f "$marker"
}

trap 'stop_stack; exit 0' INT TERM EXIT

while true; do
    coordinator_pid="$(pid_from_file "$coordinator_pid_file")"
    runner_pid="$(pid_from_file "$runner_pid_file")"
    if ! pid_is_live "$coordinator_pid"; then
        start_coordinator || true
        /bin/sleep 1
    fi
    if [[ -n "$(desktop_is_live)" ]]; then
        if ! pid_is_live "$runner_pid"; then
            start_runner || true
        fi
        /usr/bin/touch "$marker"
    elif pid_is_live "$runner_pid" || [[ -f "$marker" ]]; then
        stop_pid_file "$runner_pid_file"
        /bin/rm -f "$marker"
    fi
    /bin/sleep "$poll_seconds"
done
