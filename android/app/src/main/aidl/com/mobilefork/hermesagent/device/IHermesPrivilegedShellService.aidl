package com.mobilefork.hermesagent.device;

interface IHermesPrivilegedShellService {
    String runCommand(String command, int timeoutSeconds);
}
