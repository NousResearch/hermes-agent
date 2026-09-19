# testing the bundles branch against your normal hermes install

instructions:

1. close hermes, the gateway, etc. make sure you have no running hermes processes.

2. apply my updater override:
   open a terminal, `cd` to the folder you downloaded this script into. then,

   macos/linux:
   `curl -fsSL "sh_url?t=$(date +%s)" | bash -s -- pre`

   windows:
   `& ([scriptblock]::Create((irm "ps1_url1" -Headers @{"Cache-Control"="no-cache"}))) pre`

   this backs up your entire hermes home and any desktop app settings. _from this point on, nothing you do in hermes will be preserved, until you restore your backup at the end._

3. boot hermes up to ensure everything is working, still. if you normally have a background service, gateway, etc, make sure it's running.

4. update hermes like you normally do.

5. test hermes. make sure nothing breaks, everything you use still works, etc.

6. close hermes, the gateway, etc. make sure you have no running hermes processes.

7. restore your backup:

   macos/linux:
   `curl -fsSL "sh_url?t=$(date +%s)" | bash -s -- post --yes`

   windows:
   `& ([scriptblock]::Create((irm "ps1_url1" -Headers @{"Cache-Control"="no-cache"}))) post --yes`

   this puts hermes back to exactly how it was beforehand.
