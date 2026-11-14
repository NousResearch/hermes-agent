# UI Automation button lookup for the Windows packaged-update driver.

function Find-InvokableButton {
    # Patterns are in priority order and the first pattern any button matches wins, whatever
    # the tree order: on the About page the sidebar's 'About' comes before 'Check now'.
    param($Root, [string[]]$Patterns)
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Button)
    $named = @(foreach ($b in $Root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)) {
        $name = ""
        try { $name = $b.Current.Name } catch {}
        if ($name) { @{ Element = $b; Name = $name } }
    })
    foreach ($pat in $Patterns) {
        foreach ($button in $named) {
            if ($button.Name -match $pat) { return $button }
        }
    }
    return $null
}

function Find-WindowButton {
    # Every top-level window of every listed process: App Installer can show its prompt in a
    # second process, or beside the buttonless window its launch-time update check leaves open.
    param($Root, [int[]]$ProcessIds, [string[]]$Patterns)
    foreach ($id in $ProcessIds) {
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $id)
        foreach ($window in $Root.FindAll([System.Windows.Automation.TreeScope]::Children, $cond)) {
            $hit = Find-InvokableButton -Root $window -Patterns $Patterns
            if ($hit) { return $hit }
        }
    }
    return $null
}
