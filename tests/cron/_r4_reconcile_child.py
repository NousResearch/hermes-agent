from cron.executions import reconcile_delivery_projections
for _ in range(3):
    reconcile_delivery_projections()
print('reconciled')
