from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Iterable, Optional

from ..core.dependencies import check
from ..core.executor import ExecutionResult, Executor
from .base import BaseScanner, ScanFinding, ScanResult


class NmapScanner(BaseScanner):
    name = "nmap"
    description = "Port and service discovery with nmap"

    def build_command(
        self,
        target: str,
        ports: Optional[Iterable[int]] = None,
        all_ports: bool = False,
        timing: str | None = None,
        scripts: Optional[Iterable[str]] = None,
        udp: bool = False,
        os_detect: bool = False,
        aggressive: bool = False,
    ) -> str:
        parts = ["nmap", "-oX -", "-sV"]

        if all_ports:
            parts.append("-p-")
        elif ports:
            normalized_ports = ",".join(str(int(p)) for p in ports)
            parts.append(f"-p {normalized_ports}")

        if timing:
            value = timing if timing.startswith("T") else f"T{timing}"
            parts.append(f"-{value}")
        if scripts:
            parts.append(f"--script {','.join(scripts)}")
        if udp:
            parts.append("-sU")
        if os_detect:
            parts.append("-O")
        if aggressive:
            parts.append("-A")

        parts.append(target)
        return " ".join(parts)

    def parse_output(self, output: str, target: str, command: str | None = None) -> ScanResult:
        if not output.strip():
            return ScanResult(scanner_name=self.name, success=True, command=command, findings=[])

        try:
            root = ET.fromstring(output)
        except ET.ParseError as exc:
            return ScanResult(
                scanner_name=self.name,
                success=False,
                command=command,
                stderr=f"Failed to parse nmap XML output: {exc}",
                findings=[],
            )

        findings: list[ScanFinding] = []
        for host in root.findall("host"):
            status = host.find("status")
            if status is not None and status.attrib.get("state") == "down":
                continue

            for port_el in host.findall("./ports/port"):
                state_el = port_el.find("state")
                state = state_el.attrib.get("state") if state_el is not None else "open"
                if state not in {"open", "open|filtered"}:
                    continue

                port = None
                try:
                    port = int(port_el.attrib.get("portid", "0")) or None
                except ValueError:
                    port = None

                service_el = port_el.find("service")
                service_name = ""
                service_version = ""
                if service_el is not None:
                    name = service_el.attrib.get("name", "")
                    product = service_el.attrib.get("product", "")
                    version = service_el.attrib.get("version", "")
                    extrainfo = service_el.attrib.get("extrainfo", "")

                    service_name = " ".join(part for part in [product, name] if part).strip() or name or product
                    service_version = " ".join(part for part in [version, extrainfo] if part).strip()

                script_outputs = []
                for script_el in port_el.findall("script"):
                    script_id = script_el.attrib.get("id", "script")
                    script_out = script_el.attrib.get("output", "")
                    if script_out:
                        script_outputs.append(f"[{script_id}] {script_out}")

                evidence_chunks = [ET.tostring(port_el, encoding="unicode")]
                if script_outputs:
                    evidence_chunks.append("\n".join(script_outputs))

                findings.append(
                    ScanFinding(
                        target=target,
                        severity="medium",
                        title=f"Open port {port or 'unknown'}",
                        description=f"Port {port or 'unknown'} is {state}",
                        evidence="\n".join(chunk for chunk in evidence_chunks if chunk),
                        port=port,
                        service_name=service_name or None,
                        service_version=service_version or None,
                        remediation="Review service exposure and harden access.",
                    )
                )

        return ScanResult(scanner_name=self.name, success=True, command=command, findings=findings)

    def scan(self, target: str, executor: Executor | None = None, **kwargs) -> ScanResult:
        executor = executor or Executor()
        command = self.build_command(target, **{k: v for k, v in kwargs.items() if k in {"ports", "all_ports", "timing", "scripts", "udp", "os_detect", "aggressive"}})

        if isinstance(executor, Executor) and not check("nmap"):
            return ScanResult(
                scanner_name=self.name,
                success=False,
                command=command,
                stderr="nmap not installed",
                findings=[],
            )

        execution: ExecutionResult = executor.run(command)
        if not execution.success:
            return ScanResult(
                scanner_name=self.name,
                success=False,
                command=command,
                execution_time=execution.duration,
                stdout=execution.stdout,
                stderr=execution.stderr,
                exit_code=execution.exit_code,
                timeout=execution.timeout,
                findings=[],
            )

        parsed = self.parse_output(execution.stdout, target, command=command)
        parsed.execution_time = execution.duration
        parsed.stdout = execution.stdout
        parsed.stderr = execution.stderr
        parsed.exit_code = execution.exit_code
        return parsed


Scanner = NmapScanner
