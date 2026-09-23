#!/bin/bash
# Restore drill: prove a state-volume snapshot actually restores to a working NOVA.
#
#   deploy/aws/restore-drill.sh <snapshot-id> <subnet-id> <security-group-id> <instance-profile-name> <control-plane-image-uri> [ami-id]
#
# Launches a temporary instance whose second disk is created FROM the snapshot, checks the
# restored data (SQLite integrity of the work board and knowledge index, profiles, audit log),
# starts the control plane on it and asks it for health, agents, tasks and model status, then
# prints a DRILL REPORT to the serial console and powers off. The instance is created with
# shutdown-behaviour=terminate and the restored disk with DeleteOnTermination, so nothing is
# left behind — and it powers off after 60 minutes regardless.
#
# It deliberately does NOT start the gateway: a second gateway would poll the same Telegram
# bot and Slack app as the live one and fight it for messages.
#
# Read the report with:  aws ec2 get-console-output --instance-id <id> --latest --output text
set -euo pipefail

SNAPSHOT=$1 SUBNET=$2 SG=$3 PROFILE=$4 IMAGE=$5
AMI=${6:-$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query Parameter.Value --output text)}
REGISTRY=${IMAGE%%/*}

USER_DATA=$(cat <<EOF
#!/bin/bash
shutdown -h +60                      # hard stop, whatever happens below
exec > >(tee /var/log/nova-drill.log > /dev/console) 2>&1
set -x
report() { echo "DRILL REPORT: \$*"; }
dnf install -y -q docker sqlite >/dev/null && systemctl start docker
DEV=\$(lsblk -dpno NAME,TYPE | awk '\$2=="disk"{print \$1}' | grep -v nvme0n1 | head -1)
mkdir -p /var/lib/nova && mount "\$DEV" /var/lib/nova && report "restored disk \$DEV mounted"
H=/var/lib/nova/home
report "profiles: \$(ls \$H/profiles | tr '\n' ' ')"
report "bundle: \$(ls /var/lib/nova/bundle | tr '\n' ' ')"
report "kanban.db integrity: \$(sqlite3 \$H/kanban.db 'PRAGMA integrity_check;') tasks=\$(sqlite3 \$H/kanban.db 'SELECT count(*) FROM tasks;')"
report "knowledge index integrity: \$(sqlite3 \$H/nova-knowledge.db 'PRAGMA integrity_check;' 2>&1 | head -1)"
report "audit log events: \$(wc -l < \$H/nova/audit.jsonl)"
aws ecr get-login-password --region \$(cloud-init query region) | docker login --username AWS --password-stdin $REGISTRY >/dev/null
docker run -d --name nova-drill --network host -v /var/lib/nova:/var/lib/nova $IMAGE >/dev/null
for i in \$(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8787/platform/v1/health >/dev/null && break; sleep 3; done
report "health: \$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8787/platform/v1/health)"
report "agents: \$(curl -s -m 5 http://127.0.0.1:8787/platform/v1/agents | python3 -c 'import json,sys;print([(a["id"],a["in_sync"]) for a in json.load(sys.stdin)["agents"]])')"
report "tasks: \$(curl -s -m 5 http://127.0.0.1:8787/platform/v1/tasks | python3 -c 'import json,sys;d=json.load(sys.stdin);print(len(d["tasks"]),d["counts"])')"
report "model: \$(curl -s -m 5 http://127.0.0.1:8787/platform/v1/model | python3 -c 'import json,sys;print(json.load(sys.stdin)["state"])')"
report "control plane log: \$(docker logs nova-drill 2>&1 | grep -E '^tenant=' | tail -1)"
report "DONE"
sleep 20; poweroff
EOF
)

aws ec2 run-instances \
  --image-id "$AMI" --instance-type t3.small \
  --subnet-id "$SUBNET" --security-group-ids "$SG" \
  --iam-instance-profile Name="$PROFILE" \
  --instance-initiated-shutdown-behavior terminate \
  --metadata-options HttpTokens=required \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sdf\",\"Ebs\":{\"SnapshotId\":\"$SNAPSHOT\",\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nova-restore-drill},{Key=nova:purpose,Value=restore-drill}]" \
                       "ResourceType=volume,Tags=[{Key=Name,Value=nova-restore-drill},{Key=nova:purpose,Value=restore-drill}]" \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text
