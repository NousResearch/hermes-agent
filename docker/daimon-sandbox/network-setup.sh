#!/bin/bash
# network-setup.sh — Block private networks from the daimon-sandbox container.
# Run this after `docker compose up` or via a systemd service.
#
# Blocks: RFC1918 (10/8, 172.16/12, 192.168/16), link-local (169.254/16),
#         localhost (127/8), cloud metadata (169.254.169.254),
#         and the Docker host gateway.
#
# Allows: All public internet traffic on any port.

set -e

NETWORK_NAME="daimon-sandbox_daimon-net"

# Get the bridge interface for the network
NETWORK_ID=$(docker network inspect "$NETWORK_NAME" -f '{{.Id}}' 2>/dev/null | head -c 12)
if [ -z "$NETWORK_ID" ]; then
    echo "ERROR: Network $NETWORK_NAME not found. Run 'docker compose up' first."
    exit 1
fi

IFACE="br-${NETWORK_ID}"

# Verify interface exists
if ! ip link show "$IFACE" &>/dev/null; then
    echo "ERROR: Interface $IFACE not found."
    exit 1
fi

echo "Applying network rules to $IFACE ($NETWORK_NAME)..."

# Flush existing rules for this interface (idempotent re-apply)
iptables -D DOCKER-USER -i "$IFACE" -d 10.0.0.0/8 -j DROP 2>/dev/null || true
iptables -D DOCKER-USER -i "$IFACE" -d 172.16.0.0/12 -j DROP 2>/dev/null || true
iptables -D DOCKER-USER -i "$IFACE" -d 192.168.0.0/16 -j DROP 2>/dev/null || true
iptables -D DOCKER-USER -i "$IFACE" -d 169.254.0.0/16 -j DROP 2>/dev/null || true
iptables -D DOCKER-USER -i "$IFACE" -d 127.0.0.0/8 -j DROP 2>/dev/null || true

# Apply fresh rules
iptables -I DOCKER-USER -i "$IFACE" -d 10.0.0.0/8 -j DROP
iptables -I DOCKER-USER -i "$IFACE" -d 172.16.0.0/12 -j DROP
iptables -I DOCKER-USER -i "$IFACE" -d 192.168.0.0/16 -j DROP
iptables -I DOCKER-USER -i "$IFACE" -d 169.254.0.0/16 -j DROP
iptables -I DOCKER-USER -i "$IFACE" -d 127.0.0.0/8 -j DROP

# Block Docker host gateway (prevents SSRF to host services)
HOST_GW=$(docker network inspect "$NETWORK_NAME" -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null)
if [ -n "$HOST_GW" ]; then
    iptables -D DOCKER-USER -i "$IFACE" -d "$HOST_GW" -j DROP 2>/dev/null || true
    iptables -I DOCKER-USER -i "$IFACE" -d "$HOST_GW" -j DROP
    echo "  Blocked host gateway: $HOST_GW"
fi

echo "Done. Private networks blocked for $NETWORK_NAME."
