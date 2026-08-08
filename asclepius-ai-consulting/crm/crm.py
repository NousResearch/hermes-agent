#!/usr/bin/env python3
import argparse, sqlite3
from pathlib import Path
SCHEMA=Path(__file__).with_name('schema.sql').read_text()
def conn(path):
    db=sqlite3.connect(path); db.executescript(SCHEMA); return db
def add_lead(a):
    with conn(a.db) as db: db.execute('INSERT INTO leads(name,business,source_channel,status) VALUES(?,?,?,?)',(a.name,a.business,a.source,a.status)); print('Lead added')
def list_leads(a):
    with conn(a.db) as db:
        for r in db.execute('SELECT id,name,business,source_channel,status,created_at FROM leads ORDER BY id'): print(' | '.join(map(str,r)))
def update_status(a):
    with conn(a.db) as db: db.execute('UPDATE leads SET status=? WHERE id=?',(a.status,a.lead_id)); print('Status updated')
def add_engagement(a):
    closed='CURRENT_TIMESTAMP' if a.status=='closed_won' else 'NULL'
    with conn(a.db) as db: db.execute(f'INSERT INTO engagements(lead_id,tier,price,status,closed_at) VALUES(?,?,?,?,{closed})',(a.lead_id,a.tier,a.price,a.status)); print('Engagement added')
def revenue_summary(a):
    with conn(a.db) as db:
        rows=db.execute("SELECT tier, strftime('%Y-%m', COALESCE(closed_at, started_at)) month, SUM(price) FROM engagements WHERE status='closed_won' GROUP BY tier, month ORDER BY month,tier").fetchall()
        total=sum(r[2] for r in rows);
        for tier,month,amt in rows: print(f'{month} | {tier} | ${amt:.2f}')
        print(f'TOTAL | ${total:.2f}')
def main():
    p=argparse.ArgumentParser(description='Local SQLite CRM for ASCLEPIUS'); p.add_argument('--db',default='asclepius.db'); sub=p.add_subparsers(required=True)
    s=sub.add_parser('add-lead',help='Add a lead'); s.add_argument('--name',required=True); s.add_argument('--business',required=True); s.add_argument('--source',required=True); s.add_argument('--status',default='new'); s.set_defaults(func=add_lead)
    s=sub.add_parser('list-leads',help='List leads'); s.set_defaults(func=list_leads)
    s=sub.add_parser('update-status',help='Update lead status'); s.add_argument('--lead-id',type=int,required=True); s.add_argument('--status',required=True); s.set_defaults(func=update_status)
    s=sub.add_parser('add-engagement',help='Add engagement'); s.add_argument('--lead-id',type=int,required=True); s.add_argument('--tier',required=True); s.add_argument('--price',type=float,required=True); s.add_argument('--status',default='open'); s.set_defaults(func=add_engagement)
    s=sub.add_parser('revenue-summary',help='Print revenue by tier and month'); s.set_defaults(func=revenue_summary)
    a=p.parse_args(); a.func(a)
if __name__=='__main__': main()
