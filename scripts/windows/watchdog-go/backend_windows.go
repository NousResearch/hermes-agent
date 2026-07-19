//go:build windows

package main

import (
	"os/exec"
	"syscall"
)

func hideWindowsProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
}
