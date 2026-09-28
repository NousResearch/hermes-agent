# Smoke image for the litco-agent matter host. Runs the SAME install script the
# droplet image uses (deploy/host/install-host.sh --container: no Tailscale, no
# ufw, no fail2ban), then boots systemd so the real litco-agent.service unit,
# its ExecStartPre init and its ExecStop drain all run as on a droplet.
# Built by deploy/host/smoke.sh from a clean export of the working tree.
FROM ubuntu:24.04

ENV container=docker
# The container boots systemd as PID 1; a droplet already has it.
RUN apt-get -o Acquire::Retries=3 update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends systemd systemd-sysv dbus \
 && rm -rf /var/lib/apt/lists/*
COPY src/ /src/
RUN bash /src/deploy/host/install-host.sh --source /src --container \
 && systemctl enable litco-agent.service \
 && install -m 0755 /src/deploy/host/smoke/stub_model.py /usr/local/bin/litco-stub-model \
 && install -m 0644 /src/deploy/host/smoke/litco-stub-model.service /etc/systemd/system/litco-stub-model.service \
 && systemctl enable litco-stub-model.service \
 && rm -rf /src

STOPSIGNAL SIGRTMIN+3
CMD ["/sbin/init"]
