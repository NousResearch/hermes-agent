#!/usr/bin/env python3
import argparse
def calculate(hours_per_week,hourly_value,diagnostic_price=999):
    annual=hours_per_week*hourly_value*52; multiple=annual/diagnostic_price; return round(annual,2), round(multiple,1)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--hours-per-week',type=float,required=True); p.add_argument('--hourly-value',type=float,required=True); p.add_argument('--diagnostic-price',type=float,default=999); a=p.parse_args(); annual,multiple=calculate(a.hours_per_week,a.hourly_value,a.diagnostic_price); print(f'Annualized cost of problem: ${annual:.2f}'); print(f'That is {multiple}x the Diagnostic price')
if __name__=='__main__': main()
