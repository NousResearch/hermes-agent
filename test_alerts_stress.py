"""
Stress-Test Alerts — Push the limits of alert enrichment pipeline
===============================================================
Each alert is designed to test a specific edge case or scenario:

01. ISP BGP transit failure (critical border router, dual ISP — but only 1 neighbor down)
02. Core router power PSU failure with single PSU (SPOF on critical device)
03. Access switch down — VM hosts at risk behind it (blast radius)
04. Unknown device alert — not in NetBox at all
05. Interface flap on distribution switch with many uplinks (high noise)
06. Temperature threshold exceeded on edge router
07. Memory alert on VM host (escalated to high due to critical tag)
08. Multiple BGP neighbors down simultaneously (possible upstream outage)
09. Cable/degraded link on core switch (redundancy still ok)
10. Cactus-level: entire DC1 spine switch fails (max blast radius)

Run with:
  python test_alerts_stress.py [--dry-run] [--telegram]
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from alert_processor import enrich_alert_from_dict


ALERTS = [

    # ── 01. BGP ISP NEIGHBOR DOWN ──────────────────────────────────────────────
    # Tests: border router + ISP circuit flagging + SPOF detection + BGP alert type
    {
        "alert_id": "STRESS-01-BGP-ISP",
        "device": "DC1-BORDER-01",
        "type": "bgp-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T06:00:00+08:00",
        "description": "BGP neighbor 203.0.113.1 (Singtel-Transit) state active -> idle. Reason: hold timer expired. This is the primary transit session for Singapore DC.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-BORDER-01",
        "netbox_context": {
            "role": "Border Router",
            "device_type": "ASR 9000",
            "connected_to": ["DC1-SPINE-01", "DC1-SPINE-02"],
            "isp_circuit": "Singtel-Transit-CIR-2024-0001",
            "isp_noc": "+65-6123-4567"
        },
        "netbox_impact": {
            "redundancy": "DEGRADED — Singtel transit down, Telia still active",
            "recommended_action": "Check Singtel circuit status. If circuit issue, open with Singtel NOC."
        },
        "metrics": {
            "bgp_state": "idle",
            "neighbor": "203.0.113.1",
            "prefixes_received": 18500,
            "uptime": "3h 22m"
        }
    },

    # ── 02. PSU/POWER FAILURE — SINGLE PSU ON CRITICAL DEVICE ─────────────────
    # Tests: power-alert + PSU redundancy check + SPOF detection
    {
        "alert_id": "STRESS-02-POWER-PSU",
        "device": "DC1-CORE-RTR-01",
        "type": "power-alert",
        "severity": "high",
        "timestamp": "2026-04-19T06:05:00+08:00",
        "description": "Power supply PS1 failure detected on core router DC1-CORE-RTR-01. PS1 offline. Device is running on single PSU PS0. Fan intake temperature rising.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-CORE-RTR-01",
        "netbox_context": {
            "role": "Core Router",
            "device_type": "MX204",
            "connected_to": ["DC1-SPINE-01", "DC1-SPINE-02", "DC1-AGG-01"],
            "customer": ["DBS Bank SG", "AXS Bank"],
        },
        "netbox_impact": {
            "redundancy": "SINGLE POINT OF FAILURE — No power redundancy until PS1 replaced",
            "recommended_action": "Order replacement PS1 within 4h. Schedule maintenance window."
        },
        "netbox_cable": {
            "label": "PS1-CABLE-A",
            "type": "power"
        },
        "metrics": {
            "ps0_status": "active",
            "ps1_status": "offline",
            "power_draw_watts": 380,
            "intake_temp_c": 42
        }
    },

    # ── 03. ACCESS SWITCH DOWN — VM HOSTS AT RISK ─────────────────────────────
    # Tests: device-down-alert + VM blast radius detection
    {
        "alert_id": "STRESS-03-VM-BLAST",
        "device": "DC1-ACCESS-03",
        "type": "device-down-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T06:10:00+08:00",
        "description": "Device DC1-ACCESS-03 is unreachable. ICMP timeout. Last seen 2026-04-19T06:08:22+08:00. This switch serves the KVM cluster for the DBS Bank application tier.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-ACCESS-03",
        "netbox_context": {
            "role": "Access Switch",
            "device_type": "N9K-C93180YC-EX",
            "connected_to": ["DC1-AGG-01", "DC1-AGG-02"],
            "customer": "DBS Bank"
        },
        "netbox_impact": {
            "redundancy": "PARTIAL — Dual uplinks to agg switches, but device itself is down",
            "recommended_action": "Check physical layer first. Then check power to switch."
        },
        "metrics": {
            "icmp_rtt_ms": "timeout",
            "last_response": "2026-04-19T06:08:22+08:00",
            "cpu_last_seen": "12%"
        }
    },

    # ── 04. UNKNOWN DEVICE — NOT IN NETBOX ────────────────────────────────────
    # Tests: graceful degradation when device not found
    {
        "alert_id": "STRESS-04-UNKNOWN-DEVICE",
        "device": "SG-MGMT-FW-99",
        "type": "firewall-alert",
        "severity": "high",
        "timestamp": "2026-04-19T06:15:00+08:00",
        "description": "High CPU on firewall SG-MGMT-FW-99. CPU at 94% for last 5 minutes. Potential DDoS or configuration issue. Firewall is managing VPN concentrator for AXS Bank HQ.",
        "site": "Unknown",
        "netbox_device": "SG-MGMT-FW-99",
        "metrics": {
            "cpu": "94%",
            "mem": "78%",
            "active_connections": 480221
        }
    },

    # ── 05. INTERFACE FLAP — DISTRIBUTION SWITCH ──────────────────────────────
    # Tests: interface-alert detection + many uplinks context
    {
        "alert_id": "STRESS-05-IFACE-FLAP",
        "device": "DC1-DIST-02",
        "type": "interface-alert",
        "severity": "high",
        "timestamp": "2026-04-19T06:20:00+08:00",
        "description": "Interface GigabitEthernet1/0/3 on DC1-DIST-02 is flapping. State changes: up(06:17:01) -> down(06:17:43) -> up(06:18:12) -> down(06:19:58). This is the access layer uplink for DC1-ACCESS-04.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-DIST-02",
        "netbox_context": {
            "role": "Distribution Switch",
            "device_type": "N9K-C93108TC-EX",
            "connected_to": ["DC1-CORE-RTR-01", "DC1-CORE-RTR-02", "DC1-ACCESS-01", "DC1-ACCESS-02", "DC1-ACCESS-03", "DC1-ACCESS-04"]
        },
        "netbox_impact": {
            "redundancy": "OTHER VLANS STILL UP — Single access switch temporarily isolated",
            "recommended_action": "Check interface cable and SFP. Replace if errors persist."
        },
        "metrics": {
            "interface": "Gi1/0/3",
            "state_changes": 4,
            "last_state": "down",
            "crc_errors": 12,
            "input_errors": 47
        }
    },

    # ── 06. TEMPERATURE ALERT — EDGE ROUTER ──────────────────────────────────
    # Tests: environment-alert + thermal detection
    {
        "alert_id": "STRESS-06-TEMP",
        "device": "DC1-EDGE-RTR-01",
        "type": "environment-alert",
        "severity": "warning",
        "timestamp": "2026-04-19T06:25:00+08:00",
        "description": "Thermal threshold exceeded on DC1-EDGE-RTR-01. Intake temperature at 52°C (threshold: 50°C). Exhaust at 68°C. Fan FRU-1 is reporting degraded performance.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-EDGE-RTR-01",
        "netbox_context": {
            "role": "Edge Router",
            "device_type": "ISR4451",
            "connected_to": ["DC1-DIST-01"]
        },
        "netbox_impact": {
            "redundancy": "OK — Temperature is a warning, not a failure",
            "recommended_action": "Check room cooling. Replace fan FRU-1 at next maintenance window."
        },
        "metrics": {
            "intake_temp_c": 52,
            "exhaust_temp_c": 68,
            "fan_frut_1_status": "degraded",
            "uptime": "180d 14h"
        }
    },

    # ── 07. MEMORY ALERT ON VM HOST — CRITICAL TAG ESCALATION ────────────────
    # Tests: memory-alert detection + critical tag escalation to high
    {
        "alert_id": "STRESS-07-MEM-VM",
        "device": "DC1-VM-HOST-02",
        "type": "memory-alert",
        "severity": "average",      # would be escalated by critical tag
        "timestamp": "2026-04-19T06:30:00+08:00",
        "description": "Memory utilization at 91% on DC1-VM-HOST-02 (KVM). Running 14 VMs including DBS-Payment-API-04 and AXS-Transaction-02. Swapping detected on host. VM performance degraded.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-VM-HOST-02",
        "netbox_context": {
            "role": "VM Host",
            "device_type": "Dell PowerEdge R750",
            "connected_to": ["DC1-ACCESS-01"],
            "customer": ["DBS Bank", "AXS Bank"]
        },
        "netbox_impact": {
            "redundancy": "NO REDUNDANCY FOR VM HOST — If this host fails, 14 VMs go down",
            "recommended_action": "Migrate non-critical VMs to DC1-VM-HOST-01. Investigate memory leak."
        },
        "metrics": {
            "memory_used_percent": 91,
            "memory_total_gb": 256,
            "memory_used_gb": 233,
            "swap_used_gb": 48,
            "vm_count": 14,
            "vm_names": ["DBS-Payment-API-04", "AXS-Transaction-02", "DBS-Auth-01", "AXS-Portal-03", "DBS-Reporting-02", "TEST-Dev-Env-01", "TEST-QA-Env-02", "MGMT-VCenter-01", "MGMT-PRTG-01", "MGMT-01", "LOG-Elastic-03", "LOG-SIEM-01", "MON-Zabbix-02", "BACKUP-VEEAM-01"]
        }
    },

    # ── 08. MULTIPLE BGP NEIGHBORS DOWN — UPSTREAM OUTAGE ───────────────────
    # Tests: multi-alert correlation scenario, BGP alert with full context
    {
        "alert_id": "STRESS-08-MULTI-BGP",
        "device": "DC2-BORDER-01",
        "type": "bgp-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T06:35:00+08:00",
        "description": "BGP session down for all neighbors on DC2-BORDER-01. NTT-Transit (203.0.113.5) and PCCW-Global (203.0.113.9) both in idle state. Local preference collapsed. All outbound traffic from DC2 is down.",
        "site": "DC2 - Singapore",
        "netbox_device": "DC2-BORDER-01",
        "netbox_context": {
            "role": "Border Router",
            "device_type": "ASR 9904",
            "connected_to": ["DC2-SPINE-01", "DC2-SPINE-02"],
            "isp_circuit": "NTT-Transit-CIR-2024-0003, PCCW-Global-CIR-2024-0007",
            "isp_noc": "+65-6800-1234"
        },
        "netbox_impact": {
            "redundancy": "TOTAL FAILURE — Both transit providers down from DC2",
            "recommended_action": "This is likely a DC2 site upstream issue. Engage DC2 NOC immediately. Fail over to DC1."
        },
        "metrics": {
            "neighbor_count": 2,
            "neighbor_ntt_state": "idle",
            "neighbor_pccw_state": "idle",
            "prefixes_received_ntt": 0,
            "prefixes_received_pccw": 0,
            "local_pref": 0
        }
    },

    # ── 09. DEGRADED CABLE ON CORE SWITCH — STILL OPERATIONAL ─────────────────
    # Tests: link-alert + cable status + redundancy still ok
    {
        "alert_id": "STRESS-09-CABLE-DEGRADED",
        "device": "DC1-CORE-SW-01",
        "type": "link-alert",
        "severity": "warning",
        "timestamp": "2026-04-19T06:40:00+08:00",
        "description": "Degraded cable detected on DC1-CORE-SW-01 interface Eth1/2 (to DC1-CORE-SW-02). CRC errors: 14,000 in last 5 minutes. Interface still up but operating at reduced MTU. This is the core ISL for the SG-DC1 ring.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-CORE-SW-01",
        "netbox_context": {
            "role": "Core Switch",
            "device_type": "N9K-C93360YC-FX2",
            "connected_to": ["DC1-CORE-SW-02", "DC1-AGG-01", "DC1-AGG-02"]
        },
        "netbox_impact": {
            "redundancy": "OK — Second ISL to DC1-CORE-SW-02 still active",
            "recommended_action": "Schedule cable replacement within 24h. Monitor CRC error rate."
        },
        "netbox_cable": {
            "label": "CORE-ISL-01",
            "type": "fiber"
        },
        "metrics": {
            "interface": "Eth1/2",
            "crc_errors": 14000,
            "input_errors": 2300,
            "link_state": "up",
            "speed_actual_gbps": 10,
            "speed_configured_gbps": 100
        }
    },

    # ── 10. SPINE SWITCH FAILURE — MAXIMUM BLAST RADIUS ───────────────────────
    # Tests: worst-case scenario, spine is aggregation point, many devices behind it
    {
        "alert_id": "STRESS-10-SPINE-FAIL",
        "device": "DC1-SPINE-01",
        "type": "device-down-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T06:45:00+08:00",
        "description": "CRITICAL: DC1-SPINE-01 is down. Device unreachable since 06:44:11. This spine switch serves 6 distribution switches, 18 access switches, and 4 VM hosts. All tenant traffic for DBS Bank, AXS Bank, and PCCW Global is routing through the secondary spine DC1-SPINE-02.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-SPINE-01",
        "netbox_context": {
            "role": "Spine Switch",
            "device_type": "N9K-C9336C-FX2",
            "connected_to": ["DC1-BORDER-01", "DC1-BORDER-02", "DC1-CORE-SW-01", "DC1-CORE-SW-02", "DC1-AGG-01", "DC1-AGG-02", "DC1-AGG-03", "DC1-AGG-04", "DC1-AGG-05", "DC1-AGG-06"],
            "customer": ["DBS Bank", "AXS Bank", "PCCW Global", "NTT Ltd", "Telia Carrier", "ST Telemedia"]
        },
        "netbox_impact": {
            "redundancy": "DC1-SPINE-02 is handling failover — but at reduced capacity. If SPINE-02 fails, entire DC1 is dark.",
            "recommended_action": "IMMEDIATE: Verify DC1-SPINE-02 can handle full load. Begin investigation of SPINE-01 failure cause."
        },
        "metrics": {
            "icmp_result": "unreachable",
            "last_seen": "2026-04-19T06:44:11+08:00",
            "spine_02_cpu_percent": 78,
            "spine_02_memory_percent": 71,
            "vlans_affected": 42
        }
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # NEW ALERTS 11–22 — PUSHING THE REAL LIMITS
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 11. CASCADING SPINE FAILURE — DC1 DARK ────────────────────────────────
    # Tests: cascading failure detection — SPINE-02 goes down 30s after SPINE-01
    # Real worst-case: both spines dead = full DC1 blackout. Triggers total evacuation.
    {
        "alert_id": "STRESS-11-CASCADING-SPINES",
        "device": "DC1-SPINE-02",
        "type": "device-down-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T06:45:45+08:00",
        "description": "CRITICAL — CASCADING FAILURE: DC1-SPINE-02 is now down. Device unreachable since 06:45:40+08:00. This occurs 39 seconds after DC1-SPINE-01 went down. Both spines are now offline — DC1 is completely dark. All tenants (DBS Bank, AXS Bank, PCCW Global) have zero connectivity from this DC. Emergency failover to DC2 required.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-SPINE-02",
        "netbox_context": {
            "role": "Spine Switch",
            "device_type": "N9K-C9336C-FX2",
            "connected_to": ["DC1-BORDER-01", "DC1-BORDER-02", "DC1-CORE-SW-01", "DC1-CORE-SW-02", "DC1-AGG-01", "DC1-AGG-02", "DC1-AGG-03", "DC1-AGG-04", "DC1-AGG-05", "DC1-AGG-06"],
            "customer": ["DBS Bank", "AXS Bank", "PCCW Global", "NTT Ltd", "Telia Carrier", "ST Telemedia"]
        },
        "netbox_impact": {
            "redundancy": "TOTAL FAILURE — Both DC1 spines down. DC1 is fully isolated. DC2 must absorb all traffic.",
            "recommended_action": "ACTIVATE DC DR FAILOVER NOW. Open incident for full DC1 outage. Notify all affected customers."
        },
        "related_alerts": ["STRESS-10-SPINE-FAIL"],
        "metrics": {
            "icmp_result": "unreachable",
            "last_seen": "2026-04-19T06:45:40+08:00",
            "dc1_spines_total": 2,
            "dc1_spines_online": 0,
            "vlans_affected": 42,
            "tenant_count": 6
        }
    },

    # ── 12. BGP SESSION FLAPPING — NOT DOWN, JUST UNSTABLE ──────────────────
    # Tests: partial degradation discrimination — flapping != down
    # Border router BGP session oscillating established↔idle, prefixes unstable
    {
        "alert_id": "STRESS-12-BGP-FLAP",
        "device": "DC1-BORDER-01",
        "type": "bgp-alert",
        "severity": "high",
        "timestamp": "2026-04-19T06:50:00+08:00",
        "description": "BGP session with NTT-Transit (203.0.113.5) on DC1-BORDER-01 is flapping. State oscillation: established(06:47) -> idle(06:48) -> established(06:49) -> idle(06:50). Received prefixes fluctuating between 18,200 and 0. Transit is unstable but not fully down.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-BORDER-01",
        "netbox_context": {
            "role": "Border Router",
            "device_type": "ASR 9000",
            "connected_to": ["DC1-SPINE-01", "DC1-SPINE-02"],
            "isp_circuit": "NTT-Transit-CIR-2024-0003",
            "isp_noc": "+65-6800-1234"
        },
        "netbox_impact": {
            "redundancy": "DEGRADED — Flapping transit, Singtel still established. Traffic Engineering may be affected.",
            "recommended_action": "Monitor NTT session stability. If flap frequency increases, open with NTT NOC preemptively."
        },
        "metrics": {
            "neighbor": "203.0.113.5",
            "bgp_state": "flapping",
            "state_changes_last_15min": 8,
            "prefixes_received_min": 0,
            "prefixes_received_max": 18200,
            "uptime": "0h 3m"
        }
    },

    # ── 13. CROSS-DC INTERLINK FAILURE — GEO-REDUNDANCY EXHAUSTED ───────────
    # Tests: stretch VLAN across DC1↔DC2, geo-redundancy topology
    # The dedicated Inter-DC link fails — DC2 can reach upstream but not DC1
    {
        "alert_id": "STRESS-13-INTERDC-LINK-FAIL",
        "device": "DC1-INTERDC-01",
        "type": "link-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T06:55:00+08:00",
        "description": "Inter-DC link between DC1 and DC2 is down. Interface Eth1/0/1 on DC1-INTERDC-01 (to DC2-INTERDC-01) is offline. Stretch VLANs 100-110 (DBS Bank, AXS Bank) can no longer span both sites. DC2 is isolated from DC1 storage and backup replication.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-INTERDC-01",
        "netbox_context": {
            "role": "Inter-DC Router",
            "device_type": "NCS 5500",
            "connected_to": ["DC1-SPINE-01", "DC1-SPINE-02", "DC2-INTERDC-01"]
        },
        "netbox_impact": {
            "redundancy": "PARTIAL — DC2 can still reach upstream ISPs, but DC-to-DC replication is broken",
            "recommended_action": "Check fiber patch panel for cross-connect failure. If physical, engage DC facility team."
        },
        "netbox_cable": {
            "label": "DC1-DC2-INTERLINK-A",
            "type": "fiber"
        },
        "metrics": {
            "interface": "Eth1/0/1",
            "state": "down",
            "remote_device": "DC2-INTERDC-01",
            "vlans_affected": "100-110",
            "replication_status": "failed"
        }
    },

    # ── 14. COOLING FAILURE → THERMAL SPIRAL → PROTECTED SHUTDOWN ────────────
    # Tests: environmental → device protection shutdown cascade
    # Rack-level cooling unit fails, temp rising, devices triggering thermal shutdown
    {
        "alert_id": "STRESS-14-COOLING-FAIL",
        "device": "DC1-RACK-R7-INLET-TEMP",
        "type": "environment-alert",
        "severity": "high",
        "timestamp": "2026-04-19T07:00:00+08:00",
        "description": "Rack R7 cooling unit CU-07 has failed. Inlet temperature in rack R7 has risen from 24°C to 44°C in 12 minutes and is climbing at +2°C/min. Devices DC1-AGG-03, DC1-AGG-04, and DC1-VM-HOST-03 are approaching thermal thresholds. If inlet hits 50°C, Juniper devices will trigger thermal protection shutdown.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-RACK-R7",
        "netbox_context": {
            "role": "Rack PDU / Environmental",
            "device_type": "APC Rack PDU 2",
            "connected_to": ["DC1-AGG-03", "DC1-AGG-04", "DC1-VM-HOST-03"]
        },
        "netbox_impact": {
            "redundancy": "CRITICAL — Cooling loss in rack with 2 agg switches and 1 VM host. Thermal shutdown will cause cascading outage.",
            "recommended_action": "Dispatch facilities to replace CU-07 immediately. If temp crosses 48°C, begin graceful VM evacuation."
        },
        "metrics": {
            "inlet_temp_c": 44,
            "temp_rise_rate_c_per_min": 2,
            "threshold_c": 50,
            "cooling_unit_status": "failed",
            "devices_at_risk": ["DC1-AGG-03", "DC1-AGG-04", "DC1-VM-HOST-03"],
            "estimated_shutdown_if_no_action": "8 minutes"
        }
    },

    # ── 15. ACL CHANGE CAUSES ASYMMETRIC ROUTING — NO DEVICE DOWN ───────────
    # Tests: non-device-failure scenario — policy change creates routing anomaly
    # Firewall ACL change causes asymmetric path, packets being dropped
    {
        "alert_id": "STRESS-15-ACL-ASYM-ROUTING",
        "device": "DC1-MGMT-FW-01",
        "type": "firewall-alert",
        "severity": "high",
        "timestamp": "2026-04-19T07:05:00+08:00",
        "description": "Asymmetric routing detected on DC1-MGMT-FW-01 after an ACL change deployed at 07:00. Outbound traffic from 10.200.0.0/16 is taking a different return path than inbound. Session table showing 47% of DBS Bank sessions being silently dropped. No devices are down — this is purely a policy/routing asymmetry issue.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-MGMT-FW-01",
        "netbox_context": {
            "role": "Management Firewall",
            "device_type": "FortiGate 600E",
            "connected_to": ["DC1-BORDER-01"],
            "customer": ["DBS Bank", "AXS Bank"]
        },
        "netbox_impact": {
            "redundancy": "OK — No redundancy impact, but policy issue is dropping production traffic",
            "recommended_action": "Review ACL change log at 07:00. Most likely cause: route-map or ACL change on upstream peer altered return path. Roll back ACL or adjust asymmetric routing policy."
        },
        "metrics": {
            "session_table_hit_rate_percent": 47,
            "policy_change_time": "2026-04-19T07:00:00+08:00",
            "affected_subnet": "10.200.0.0/16",
            "dropped_sessions": 12847,
            "active_sessions": 14320
        }
    },

    # ── 16. ISP PEER ROUTER FAILURE — 3 BGP SESSIONS DOWN FROM ONE DEVICE ────
    # Tests: single ISP node failure affecting multiple sessions on our border
    # NTT's peer router failed, affecting all 3 of our sessions via that peer
    {
        "alert_id": "STRESS-16-ISP-PEER-FAIL",
        "device": "DC1-BORDER-01",
        "type": "bgp-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T07:10:00+08:00",
        "description": "Three BGP sessions on DC1-BORDER-01 have simultaneously gone idle: NTT-Transit-1 (203.0.113.5), NTT-Transit-2 (203.0.113.9), and NTT-Peering (203.0.113.13). All three are NTT upstream peers. Investigation shows NTT's edge router ntt-sg-dc1-pe01 is experiencing a control plane issue. Our Singtel and PCCW sessions remain established. NTT is the primary transit — 60% of traffic affected.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-BORDER-01",
        "netbox_context": {
            "role": "Border Router",
            "device_type": "ASR 9000",
            "connected_to": ["DC1-SPINE-01", "DC1-SPINE-02"],
            "isp_circuit": "NTT-Transit-CIR-2024-0003, NTT-Peering-CIR-2024-0004",
            "isp_noc": "+65-6800-1234"
        },
        "netbox_impact": {
            "redundancy": "DEGRADED — NTT sessions all down, Singtel and PCCW still up. 60% of traffic rerouted to secondary ISPs.",
            "recommended_action": "Contact NTT NOC — their peer router is the root cause. No action on our side required. Monitor rerouted traffic load on Singtel/PCCW."
        },
        "metrics": {
            "neighbors_down": 3,
            "neighbor_203.0.113.5_state": "idle",
            "neighbor_203.0.113.9_state": "idle",
            "neighbor_203.0.113.13_state": "idle",
            "traffic_affected_percent": 60,
            "isp_peer_device": "ntt-sg-dc1-pe01"
        }
    },

    # ── 17. SD-WAN TUNNEL DEGRADATION — OVERLAY/UNDERLAY ─────────────────────
    # Tests: SD-WAN overlay tunnel health, packet loss on tunneled traffic
    # DC1-to-DC2 SD-WAN tunnel degraded, 25% packet loss, latency spike
    {
        "alert_id": "STRESS-17-SDWAN-TUNNEL-DEG",
        "device": "DC1-SDWAN-01",
        "type": "tunnel-alert",
        "severity": "high",
        "timestamp": "2026-04-19T07:15:00+08:00",
        "description": "SD-WAN tunnel DC1-to-DC2 via ISP-2 (Telia) is degraded. Packet loss: 25%. Latency: 380ms (normal: 12ms). Jitter: 85ms. Tunnel DC1-SDWAN-01:TLIA-DC2 is operating in degraded mode. Business video conferencing and voice traffic is being affected. Underlay ISP connectivity is fine — issue is within the SD-WAN overlay tunnel.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-SDWAN-01",
        "netbox_context": {
            "role": "SD-WAN Edge",
            "device_type": "Cisco SD-WAN C8000",
            "connected_to": ["DC1-DIST-01"]
        },
        "netbox_impact": {
            "redundancy": "DEGRADED — SD-WAN overlay degraded, traffic is rerouting but with latency impact. Underlay is fine.",
            "recommended_action": "Check SD-WAN controller (vBond/vManage) for tunnel state. Could be encryption issue, MTU fragmentation, or ISP PE congestion on Telia path."
        },
        "metrics": {
            "tunnel_name": "DC1-SDWAN-01:TLIA-DC2",
            "packet_loss_percent": 25,
            "latency_ms": 380,
            "jitter_ms": 85,
            "normal_latency_ms": 12,
            "tunnel_state": "degraded",
            "tlia_bw_actual_mbps": 180,
            "tlia_bw_configured_mbps": 500
        }
    },

    # ── 18. STORAGE RAID DEGRADED — BACKUP JOBS FAILING ──────────────────────
    # Tests: storage alerting, RAID degraded, backup infrastructure
    # Backup storage server has degraded RAID, backup jobs for DBS Bank failing
    {
        "alert_id": "STRESS-18-STORAGE-RAID-DEG",
        "device": "DC1-BACKUP-STORE-01",
        "type": "hardware-alert",
        "severity": "high",
        "timestamp": "2026-04-19T07:20:00+08:00",
        "description": "RAID-5 degraded on DC1-BACKUP-STORE-01. Disk PD11 has failed. RAID is still operational but running on parity. Two backup jobs for DBS Bank (DBS-Full-Backup-04, DBS-Incr-Backup-07) have failed. If a second disk fails before replacement, full data loss will occur. Replacement disk PD11 is in spare parts inventory.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-BACKUP-STORE-01",
        "netbox_context": {
            "role": "Backup Storage",
            "device_type": "Dell PowerVault ME5024",
            "connected_to": ["DC1-ACCESS-01"],
            "customer": ["DBS Bank"]
        },
        "netbox_impact": {
            "redundancy": "CRITICAL — RAID degraded, one more disk failure = total backup loss for DBS. Restore capability is now single-disk dependent.",
            "recommended_action": "Replace PD11 immediately. Pause non-critical backup jobs to reduce RAID rebuild load. Verify last successful backup was within RPO."
        },
        "metrics": {
            "raid_level": "RAID-5",
            "raid_state": "degraded",
            "failed_disk": "PD11",
            "disks_total": 12,
            "disks_online": 11,
            "rebuild_status": "pending",
            "backup_jobs_failed": 2,
            "last_successful_backup": "2026-04-18T22:00:00+08:00"
        }
    },

    # ── 19. LOAD BALANCER CERT EXPIRY — 48HRS TO EXPIRY ─────────────────────
    # Tests: proactive certificate expiry detection (not a failure yet)
    # TLS cert on production LB VIP expiring in 48 hours — not down, but urgent
    {
        "alert_id": "STRESS-19-CERT-EXPIRY",
        "device": "DC1-LB-01",
        "type": "certificate-alert",
        "severity": "high",
        "timestamp": "2026-04-19T07:25:00+08:00",
        "description": "TLS certificate for VIP 203.0.114.50:443 (DBS Bank Production API) will expire in 48 hours (2026-04-21T07:25:00+08:00). Certificate CN: api.dbsbank.sg, SANs: [api.dbsbank.sg, pay.dbsbank.sg]. Certificate is still valid and service is operational — this is a proactive renewal alert, not a failure.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-LB-01",
        "netbox_context": {
            "role": "Load Balancer",
            "device_type": "F5 BIG-IP 1600",
            "connected_to": ["DC1-DIST-01"],
            "customer": ["DBS Bank"]
        },
        "netbox_impact": {
            "redundancy": "OK — Service operational NOW, but will go dark in 48h if cert is not renewed",
            "recommended_action": "Renew cert now via cert-manager or manually. F5 BIG-IP: run 'tmsh modify sys crypto cert /Common/api-dbsbank-sg restart' post-renewal."
        },
        "metrics": {
            "vip": "203.0.114.50",
            "port": 443,
            "cert_cn": "api.dbsbank.sg",
            "cert_sans": "api.dbsbank.sg, pay.dbsbank.sg",
            "expires_at": "2026-04-21T07:25:00+08:00",
            "hours_until_expiry": 48,
            "cert_issuer": "DigiCert Inc"
        }
    },

    # ── 20. TWO VM HOSTS DOWN SIMULTANEOUSLY — MAINTENANCE MISSTEP ──────────
    # Tests: VM redundancy exhaustion, bulk VM evacuation scenario
    # During a maintenance window, DC1-VM-HOST-01 and DC1-VM-HOST-02 both went down
    {
        "alert_id": "STRESS-20-DOUBLE-VM-HOST",
        "device": "DC1-VM-HOST-01",
        "type": "device-down-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T07:30:00+08:00",
        "description": "CRITICAL: Both DC1-VM-HOST-01 and DC1-VM-HOST-02 are simultaneously down during a scheduled maintenance window. This is NOT a redundant failure — it is a maintenance mishap. 28 VMs are offline including: DBS-Payment-API-01, DBS-Payment-API-02, DBS-Auth-01, AXS-Transaction-01, AXS-Transaction-02, and MGMT-VCenter-01. DC1-VM-HOST-03 is isolated and cannot absorb the load.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-VM-HOST-01",
        "netbox_context": {
            "role": "VM Host",
            "device_type": "Dell PowerEdge R750",
            "connected_to": ["DC1-ACCESS-01"],
            "customer": ["DBS Bank", "AXS Bank"]
        },
        "netbox_impact": {
            "redundancy": "TOTAL FAILURE — Both VM hosts down simultaneously. All VMs on these hosts are offline. VM evacuation to DC2 is the only recovery path.",
            "recommended_action": "ABORT maintenance window. Investigate why both hosts went down simultaneously. This should not happen in a properly isolated maintenance procedure. Begin emergency VM recovery to DC2."
        },
        "related_alerts": ["Maintenance-Window-DC1-VM-HOSTS"],
        "metrics": {
            "vm_hosts_total": 3,
            "vm_hosts_online": 1,
            "vm_hosts_down": 2,
            "vms_total_offline": 28,
            "critical_vms_offline": ["DBS-Payment-API-01", "DBS-Payment-API-02", "DBS-Auth-01", "AXS-Transaction-01", "AXS-Transaction-02", "MGMT-VCenter-01"],
            "vm_host_03_capacity_remaining_percent": 12
        }
    },

    # ── 21. FULL DC2 SITE ISOLATED — BGP UP BUT ROUTING BROKEN ──────────────
    # Tests: site-level BCP/DR trigger, BGP OK but internal routing broken
    # DC2 can reach upstream ISPs but internal DC2-to-DC1 routing is broken
    {
        "alert_id": "STRESS-21-DC2-ISOLATION",
        "device": "DC2-CORE-RTR-01",
        "type": "routing-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T07:35:00+08:00",
        "description": "DC2 is isolated from the rest of the network. BGP sessions to NTT (203.0.113.17) and PCCW (203.0.113.21) on DC2-CORE-RTR-01 are both established. DC2 can reach the internet. However, OSPF adjacencies with DC1 border routers have dropped. Internal routing table for DC2 shows no routes to DC1 prefix 10.0.0.0/8. DC2 is routing hairpin — traffic enters and exits via same DC2 router.",
        "site": "DC2 - Singapore",
        "netbox_device": "DC2-CORE-RTR-01",
        "netbox_context": {
            "role": "Core Router",
            "device_type": "MX204",
            "connected_to": ["DC2-SPINE-01", "DC2-SPINE-02", "DC2-BORDER-01"],
            "customer": ["DBS Bank", "AXS Bank", "PCCW Global"]
        },
        "netbox_impact": {
            "redundancy": "TOTAL SITE ISOLATION — DC2 is cut off from DC1. No east-west DC-to-DC traffic. BCP should be activated for DC2-dependent services.",
            "recommended_action": "Check OSPF area configuration on DC2-CORE-RTR-01 and DC1 border routers. Most likely cause: OSPF area was accidentally removed or interface authentication failed during a config push."
        },
        "metrics": {
            "bgp_sessions_established": 2,
            "ospf_adjacencies_to_dc1": 0,
            "dc2_default_route_via_bgp": "active",
            "dc2_internal_routing": "broken",
            "prefix_10.0.0.0/8_reachable": False,
            "dc2_upstream_connectivity": "ok"
        }
    },

    # ── 22. UPS CAPACITY < 15 MIN RUNTIME — GRACEFUL SHUTDOWN STAGING ───────
    # Tests: power emergency, staged graceful shutdown, UPS capacity alerting
    # UPS serving rack row 5 has 13 minutes of backup runtime at current load
    {
        "alert_id": "STRESS-22-UPS-LOW-RUNTIME",
        "device": "DC1-UPS-R5-MGMT",
        "type": "power-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T07:40:00+08:00",
        "description": "UPS DC1-UPS-R5-MGMT battery capacity is critically low. Runtime at current load: 13 minutes. Mains power has been lost — UPS is on battery. Load is 87% of capacity. When UPS depletes, it will hard-shutdown all connected devices in rack row 5 without graceful shutdown. Devices at risk: DC1-AGG-03, DC1-AGG-04, DC1-VM-HOST-03, DC1-BACKUP-STORE-01.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-UPS-R5",
        "netbox_context": {
            "role": "UPS / Power Infrastructure",
            "device_type": "APC Smart-UPS 5000VA",
            "connected_to": ["DC1-AGG-03", "DC1-AGG-04", "DC1-VM-HOST-03", "DC1-BACKUP-STORE-01"]
        },
        "netbox_impact": {
            "redundancy": "IMMINENT FAILURE — UPS has 13 min runtime. After that, hard shutdown of all row-5 devices = data loss and hardware damage risk.",
            "recommended_action": "IMMEDIATE: Start graceful VM evacuation from DC1-VM-HOST-03 NOW. Begin application failover for DBS/AXS services. Contact facilities to restore mains power to UPS R5 feed."
        },
        "metrics": {
            "ups_runtime_minutes": 13,
            "battery_capacity_percent": 18,
            "load_percent": 87,
            "mains_status": "lost",
            "ups_status": "on_battery",
            "devices_affected": ["DC1-AGG-03", "DC1-AGG-04", "DC1-VM-HOST-03", "DC1-BACKUP-STORE-01"],
            "estimated_shutdown": "13 minutes"
        }
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # NEW ALERTS 23–29 — Targeting NetBox gaps + real device topology
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 23. BGP FLAP — DC1-BORDER-01 SINGTEL NEIGHBOR BOUNCING ─────────────
    # Tests: BGP instability detection, ISP circuit quality flagging,
    #        LLM distinguishes "flapping" vs "single neighbor down"
    #        Uses REAL NetBox data: DC1-BORDER-01 Gi0/0/2 = Singtel ISP-01 CIR
    {
        "alert_id": "STRESS-23-BGP-FLAP",
        "device": "DC1-BORDER-01",
        "type": "bgp-alert",
        "severity": "high",
        "timestamp": "2026-04-19T08:00:00+08:00",
        "description": "BGP neighbor 203.0.113.1 (Singtel ISP-01) is flapping — session Established → Idle → Established every 3–8 minutes. This has happened 11 times in the last 2 hours. Not a full outage, but indicates an unstable circuit or CPE issue on the Singtel handoff. Each flap causes a 30-second traffic drop for affected prefixes.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-BORDER-01",
        "netbox_context": {
            "role": "Border Router",
            "device_type": "ASR 9000",
            "connected_to": ["DC1-CORE-01", "DC1-CORE-02"],
            "isp_circuit": "Singtel ISP-01 CIR-2024-0001"
        },
        "netbox_impact": {
            "redundancy": "DEGRADED — Singtel ISP-01 flapping. Telia ISP-02 still stable. Transit still available but Singtel prefixes are unstable.",
            "recommended_action": "Open with Singtel NOC (+65-6123-4567). Request circuit quality test (CRC errors, SNR). If CRC errors present on Gi0/0/2, escalate to field support for fiber polishing."
        },
        "metrics": {
            "bgp_state": "flapping",
            "neighbor": "203.0.113.1",
            "flap_count_2h": 11,
            "avg_uptime_per_session": "4m 32s",
            "prefixes_received": 18420,
            "interface": "Gi0/0/2",
            "interface_errors_5m": 47,
            "crc_errors": 312
        }
    },

    # ── 24. TRANSIT RING SPLIT — DC1-TRANSIT-RING-01 GOES DOWN ─────────────
    # Tests: Transit ring topology awareness, customer-vlan impact,
    #        LLM understands ring vs star topology
    #        Uses REAL NetBox data: DC1-TRANSIT-RING-01 connects to spines
    {
        "alert_id": "STRESS-24-TRANSIT-RING-DOWN",
        "device": "DC1-TRANSIT-RING-01",
        "type": "device-down-alert",
        "severity": "high",
        "timestamp": "2026-04-19T08:05:00+08:00",
        "description": "DC1-TRANSIT-RING-01 is unreachable. This transit switch serves customer VLAN 101 (Customer-A) for 6 downstream EX4300 access switches (DC1-TRANSIT-RING-01-DEV01 through DEV06). 14 customer-facing ports have gone offline. Customer-A (DBS Bank secondary) has lost 30% of its DC1 access ports.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-TRANSIT-RING-01",
        "netbox_context": {
            "role": "Transit Switch",
            "device_type": "EX4300",
            "connected_to": ["DC1-SPINE-01", "DC1-SPINE-02"]
        },
        "netbox_impact": {
            "redundancy": "SPLIT — Transit ring is broken at DC1-TRANSIT-RING-01. Downstream DEV01–DEV06 are isolated from spine. No redundancy path without the failed transit switch.",
            "recommended_action": "DC1-TRANSIT-RING-01 needs immediate replacement or recovery. Customer-A VLAN101 is degraded. Check if spanning-tree has blocked a redundant path."
        },
        "netbox_vlan": {
            "vlan_id": 101,
            "vlan_name": "Customer-A",
            "customer": "DBS Bank"
        },
        "metrics": {
            "ring_position": "node-1-of-5",
            "devices_downstream": 6,
            "customer_ports_offline": 14,
            "vlan_101_status": "degraded",
            "stp_state": "forwarding"
        }
    },

    # ── 25. SPINE SWITCH WITH NO NETBOX INTERFACES — CONNECTIVITY MYSTERY ──
    # Tests: what happens when device has no interface data in NetBox —
    #        pipeline must not crash, LLM must flag "insufficient data"
    #        DC1-SPINE-01 has 0 interfaces defined — this is a real data gap
    {
        "alert_id": "STRESS-25-SPINE-NO-IFACES",
        "device": "DC1-SPINE-01",
        "type": "device-down-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T08:10:00+08:00",
        "description": "DC1-SPINE-01 is unreachable. This is a spine switch — the top of the fabric. If it is truly down and cannot recover, ALL access and distribution switches in DC1 lose their uplinks. Cannot determine exact blast radius because DC1-SPINE-01 has no interfaces defined in NetBox — connectivity map is incomplete.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-SPINE-01",
        "netbox_context": {
            "role": "Spine",
            "device_type": "Spine Switch",
            "connected_to": []
        },
        "netbox_impact": {
            "redundancy": "UNKNOWN — DC1-SPINE-01 has no interface data in NetBox. Cannot trace connected devices. Assume worst case: full DC1 fabric impact until connectivity is confirmed.",
            "recommended_action": "IMMEDIATE: Physically verify DC1-SPINE-01 status. If truly down, this is a CRITICAL-1 incident. Update NetBox with interface data once device is recovered. MANUAL TRACEBACK REQUIRED — NetBox topology is incomplete."
        },
        "metrics": {
            "netbox_interfaces_defined": 0,
            "netbox_connected_devices_found": 0,
            "spine_oversubscription_ratio": "4:1",
            "fabric_type": "leaf-spine"
        }
    },

    # ── 26. VM HOST NAMING MISMATCH — DC1-VMHOST-01 (not DC1-VM-HOST-01) ───
    # Tests: naming inconsistency in NetBox — alert says DC1-VM-HOST-01
    #        but NetBox has DC1-VMHOST-01. Lookup must handle fuzzy match
    #        or gracefully return "device not found" instead of crashing.
    {
        "alert_id": "STRESS-26-VM-HOST-NAME-MISMATCH",
        "device": "DC1-VM-HOST-01",
        "type": "device-down-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T08:15:00+08:00",
        "description": "DC1-VM-HOST-01 is down. This is a critical VM host — vCenter MGMT-VCenter-01 and 6 production VMs are hosted here. vCenter cannot reach DC1-VM-HOST-01 via any management path. Note: NetBox has this device listed as DC1-VMHOST-01 (no hyphen) — there is a naming inconsistency between monitoring and NetBox.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-VM-HOST-01",
        "netbox_context": {
            "role": "VM Host",
            "device_type": "Server"
        },
        "netbox_impact": {
            "redundancy": "TOTAL FAILURE — DC1-VM-HOST-01 (NetBox: DC1-VMHOST-01) is down. All VMs evacuated or offline. VMHOST-02 still operational but cannot absorb full load.",
            "recommended_action": "REMEDIATION: Fix NetBox device name DC1-VMHOST-01 → DC1-VM-HOST-01 to match monitoring. Begin VM recovery. Check for host hardware failure (iLO/DRAC unreachable)."
        },
        "metrics": {
            "vms_on_host": 7,
            "vms_evacuated": 0,
            "vms_offline": 7,
            "critical_vms": ["MGMT-VCenter-01", "DBS-Payment-Queue-01", "AXS-Recon-Service-01"],
            "vmhost_02_capacity_percent": 78
        }
    },

    # ── 27. CASCADING MULTI-DEVICE — DC2 CORES UNREACHABLE SIMULTANEOUSLY ──
    # Tests: multi-device correlation, site-level BCP trigger,
    #        LLM recognizes "same timestamp + same site = likely upstream cause"
    #        Uses REAL NetBox devices: DC2-CORE-01, DC2-CORE-02, DC2-BORDER-01
    {
        "alert_id": "STRESS-27-DC2-CASCADE",
        "device": "DC2-CORE-01",
        "type": "device-down-alert",
        "severity": "critical",
        "timestamp": "2026-04-19T08:20:00+08:00",
        "description": "SIMULTANEOUS ALERT: DC2-CORE-01 and DC2-CORE-02 both went down at the exact same timestamp (08:20:00+08:00). DC2-BORDER-01 is still up and BGP sessions are established, but DC2 is completely isolated internally — no internal routing, no east-west traffic. This is not a coincidence — it points to a shared upstream dependency (power, fiber bundle, or shared infra below the core).",
        "site": "DC2 - Singapore",
        "netbox_device": "DC2-CORE-01",
        "netbox_context": {
            "role": "Core Switch",
            "device_type": "Nexus 9300",
            "connected_to": ["DC2-BORDER-01"]
        },
        "netbox_impact": {
            "redundancy": "TOTAL SITE FAILURE — Both DC2 core switches down simultaneously. DC2-BORDER-01 is edge-only. All internal DC2 services are offline. Activate DC2 BCP.",
            "recommended_action": "IMMEDIATE: This is NOT a core switch failure — it is a shared infrastructure failure causing both cores to go dark. Check: (1) Shared power feed to rack, (2) Fiber patch panel feeding both cores, (3) Shared upstream switch. Activate DC2 failover to DC1."
        },
        "related_alerts": ["STRESS-27-DC2-CASCADE-B", "STRESS-27-DC2-CASCADE-C"],
        "metrics": {
            "dc2_cores_total": 2,
            "dc2_cores_down": 2,
            "dc2_border_up": True,
            "dc2_bgp_sessions_established": 3,
            "dc2_internal_routing": "down",
            "downstream_devices_affected": 41,
            "shared_upstream_found": "none — investigation required"
        }
    },

    # ── 28. CUSTOMER-CRITICAL ALERT — AXS BANK TRANSACTION SERVICE DEGRADED
    # Tests: tenant/customer tagging impact, customer-specific recommended actions,
    #        severity escalation via customer tag (AXS Bank = tenant, not in device tags)
    #        Tenant exists in NetBox (AXS Bank id=1), device has no tenant assigned
    {
        "alert_id": "STRESS-28-AXS-SERVICE-DEGRADE",
        "device": "DC1-DIST-02",
        "type": "interface-alert",
        "severity": "warning",
        "timestamp": "2026-04-19T08:25:00+08:00",
        "description": "Interface GigabitEthernet1/1 on DC1-DIST-02 is err-disabled due to STP violation. This port connects to AXS Bank's primary access switch. AXS transaction processing is degraded — 40% of AXS payment transactions are failing with timeout errors. Port was err-disabled at 08:20 and has not auto-recovered.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-DIST-02",
        "netbox_context": {
            "role": "Distribution",
            "device_type": "Distribution Switch",
            "connected_to": ["DC1-SPINE-01", "DC1-SPINE-02", "AXS-BANK-ACCESS-01"]
        },
        "netbox_impact": {
            "redundancy": "PARTIAL — Single interface err-disabled on DC1-DIST-02. AXS Bank has redundant paths via secondary access switch, but 40% of transactions are failing through the degraded path.",
            "recommended_action": "IMMEDIATE for AXS Bank SLA: Clear err-disabled port on DC1-DIST-02 Gi1/1 ('switchport port-security violation restrict' under config). Open AXS Bank incident ticket. If STP loop is real, find and remove the loop before re-enabling port."
        },
        "netbox_vlan": {
            "vlan_id": 101,
            "vlan_name": "Customer-A",
            "customer": "AXS Bank"
        },
        "metrics": {
            "interface": "Gi1/1",
            "interface_status": "err-disabled",
            "stp_reason": "port-security violation",
            "axs_transaction_success_rate": 60,
            "axs_transactions_failing_per_min": 240,
            "auto_recovery_attempted": False
        }
    },

    # ── 29. NTP REFLECTOR DDoS — FIREWALL ALERT WITH IMPRECISE DEVICE ─────
    # Tests: NTP-alert type (new), imprecision in device identification
    #        Alert says 'DC1-MGMT-FW-01' but description is actually about a
    #        policy/session issue on the firewall — type detection must be accurate
    #        Device IS in NetBox (DC1-MGMT-FW-01, id=62, role Mgmt Firewall)
    {
        "alert_id": "STRESS-29-NTP-REFLECTOR",
        "device": "DC1-MGMT-FW-01",
        "type": "ntp-alert",
        "severity": "high",
        "timestamp": "2026-04-19T08:30:00+08:00",
        "description": "DC1-MGMT-FW-01 is receiving a high volume of NTP reply packets (47,000 pps) from source 10.255.0.0/16 — this is an NTP reflection/amplification attack. The firewall's session table shows 18,000 half-open NTP sessions. CPU on DC1-MGMT-FW-01 is at 89%. NTP server function is impaired — all devices sync'ing to DC1-MGMT-FW-01 for time are experiencing clock drift.",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-MGMT-FW-01",
        "netbox_context": {
            "role": "Mgmt Firewall",
            "device_type": "Firewall",
            "connected_to": ["DC1-CORE-01", "DC1-CORE-02"]
        },
        "netbox_impact": {
            "redundancy": "DEGRADED — NTP reflection attack saturating DC1-MGMT-FW-01. NTP service is impaired. Clock drift may cause: (1) SAML/OIDC token issues, (2) log correlation failures, (3) PKI certificate validation errors.",
            "recommended_action": "URGENT: Block NTP traffic from source 10.255.0.0/16 on DC1-MGMT-FW-01 ACL. Null-route 10.255.0.0/16 at border. Investigate 10.255.0.0/16 — this is an internal source, so compromised internal host is running NTP amplification tool. Quarantine source host."
        },
        "metrics": {
            "ntp_reply_pps": 47000,
            "ntp_session_count": 18000,
            "fw_cpu_percent": 89,
            "source_subnet": "10.255.0.0/16",
            "ntp_mode": 7,
            "amplification_ratio": "556x",
            "affected_ntp_clients": 312
        }
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # END OF NEW ALERTS 23–29
    # ═══════════════════════════════════════════════════════════════════════════
]


def run_tests(dry_run: bool = True, telegram: bool = False):
    print(f"\n{'='*60}")
    print(f"  STRESS TEST — {len(ALERTS)} ALERTS")
    print(f"  Mode: {'DRY RUN (prompt only)' if dry_run else 'LIVE (Telegram)' if telegram else 'LIVE (no Telegram)'}")
    print(f"{'='*60}\n")

    for i, alert in enumerate(ALERTS, 1):
        alert_id = alert.get("alert_id", f"alert-{i}")
        device = alert.get("device", "unknown")
        alert_type = alert.get("type", "unknown")
        severity = alert.get("severity", "average")
        description = alert.get("description", "")[:80]

        print(f"[{i:02d}/{len(ALERTS)}] {alert_id}")
        print(f"       Device: {device} | Type: {alert_type} | Severity: {severity}")
        print(f"       {description}...")

        if dry_run:
            from alert_processor import build_enrichment_prompt, AlertRecord
            record = AlertRecord.from_mock_lab(alert)
            # Mock NetBox lookup results
            # Build a proper mock nb_device matching real NetBox structure
            nb_ctx = alert.get("netbox_context", {})
            nb_device = {
                "id": 1,
                "name": device,
                "role": {"display": nb_ctx.get("role", "Unknown"), "slug": nb_ctx.get("role", "unknown").lower().replace(" ", "-"), "name": nb_ctx.get("role", "Unknown")},
                "site": {"display": alert.get("site", "Unknown"), "name": alert.get("site", "Unknown")},
                "device_type": {"display": nb_ctx.get("device_type", ""), "model": nb_ctx.get("device_type", "")},
                "status": {"value": "active"},
                "primary_ip": {"address": ""},
                "tags": [{"slug": "critical"}] if "DBS Bank" in str(nb_ctx.get("customer", "")) else [],
                "serial": "N/A"
            }
            connected = [{"device_name": "peer-" + str(i), "interface": "Eth0/0", "device_role": "switch"}]
            cables = []
            if alert.get("netbox_cable"):
                cables = [alert["netbox_cable"]]
            prompt = build_enrichment_prompt(record, nb_device, connected, cables, [])
            print(f"       Prompt length: {len(prompt)} chars")
            print(f"       {'─'*54}")
            print(prompt[:800])
            print("...")
            print(f"       {'─'*54}")
        else:
            result = enrich_alert_from_dict(alert)
            print(f"       Result: {result.get('status', '?')} | Sent: {result.get('sent', '?')}")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress-test alert enrichment")
    parser.add_argument("--live", action="store_true", help="Run live (send to Telegram)")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts only (default)")
    args = parser.parse_args()

    run_tests(dry_run=not args.live, telegram=args.live)
