export declare const style: {
    layout: (t: import("..").Traits) => {
        shape: string;
        draw: ((b: import("./shapes").Body) => string) | undefined;
        body: import("./shapes").Body;
        face: import("./shapes").Ellipse;
        petals: {
            cx: number;
            cy: number;
            r: number;
        }[];
        extra: string[];
        eyes: import("./compose").Eye[];
    };
    render: (l: ReturnType<(t: import("..").Traits) => {
        shape: string;
        draw: ((b: import("./shapes").Body) => string) | undefined;
        body: import("./shapes").Body;
        face: import("./shapes").Ellipse;
        petals: {
            cx: number;
            cy: number;
            r: number;
        }[];
        extra: string[];
        eyes: import("./compose").Eye[];
    }>, p: import("..").Palette, mo?: boolean) => string;
    background: false;
};
//# sourceMappingURL=blob.d.ts.map